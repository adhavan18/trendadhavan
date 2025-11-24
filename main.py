# main.py
import os
import time
import json
import random
import logging
import subprocess
from datetime import datetime, timedelta

import requests
import tweepy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Config from env ---
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_KEY_SECRET = os.getenv("TWITTER_API_KEY_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
HF_TOKEN = os.getenv("HF_TOKEN")  # HuggingFace inference token
YOUR_USERNAME = os.getenv("YOUR_USERNAME")
YOUR_USER_ID = os.getenv("YOUR_USER_ID")
MAX_REPLIES_PER_DAY = int(os.getenv("MAX_REPLIES_PER_DAY", "5"))  # safety cap
NICHE_KEYWORDS = os.getenv("NICHE_KEYWORDS", "AI,startup,business,finance,politics").split(",")
HF_MODEL = os.getenv("HF_MODEL", "gpt2")
CONTROVERSY_SLIDER = int(os.getenv("CONTROVERSY_SLIDER", "2"))  # 0 (safe) to 10 (risky)

# state file in repo to keep daily count (will be committed back)
STATE_FILE = "state.json"

# --- Twitter clients ---
# OAuth1 for v1.1 endpoints (trends), and v2 client for searching & posting
auth1 = tweepy.OAuth1UserHandler(
    TWITTER_API_KEY, TWITTER_API_KEY_SECRET,
    TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
)
api_v1 = tweepy.API(auth1, wait_on_rate_limit=True)

client_v2 = tweepy.Client(
    consumer_key=TWITTER_API_KEY,
    consumer_secret=TWITTER_API_KEY_SECRET,
    access_token=TWITTER_ACCESS_TOKEN,
    access_token_secret=TWITTER_ACCESS_SECRET,
    wait_on_rate_limit=True
)

# --- Helpers: state management & commit back to repo (via GITHUB_TOKEN in workflow) ---
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    # default state
    return {"date": datetime.utcnow().date().isoformat(), "replies_today": 0}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)
    # commit back to repo so daily counts persist across runs (workflow will push)
    try:
        subprocess.run(["git", "config", "user.email", "action@github.com"], check=True)
        subprocess.run(["git", "config", "user.name", "github-action"], check=True)
        subprocess.run(["git", "add", STATE_FILE], check=True)
        subprocess.run(["git", "commit", "-m", f"update state: {state}"], check=True)
        # push using GITHUB_TOKEN set in workflow (remote preconfigured in workflow)
        repo_remote = os.getenv("REPO_REMOTE")
        if repo_remote:
            subprocess.run(["git", "push", repo_remote, "HEAD:main"], check=True)
    except subprocess.CalledProcessError as e:
        logger.warning("Could not commit state back to repo: %s", e)

# --- HF text generation (inference API) ---
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def generate_reply_hf(tweet_text, style_examples, max_tokens=80):
    prompt = (
        f"Style examples from @{YOUR_USERNAME}: {style_examples}\n\n"
        f"Tweet: {tweet_text}\nReply in the user's style, concise (<=280 chars). Add value or a question."
    )
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": 0.7}}
    try:
        r = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=20)
        if r.status_code == 200:
            j = r.json()
            # model outputs differ: try multiple shapes
            if isinstance(j, list) and "generated_text" in j[0]:
                out = j[0]["generated_text"]
            elif isinstance(j, dict) and "generated_text" in j:
                out = j["generated_text"]
            else:
                # some models return string
                out = j[0].get("generated_text", "") if isinstance(j, list) else str(j)
            # attempt to extract the reply (remove prompt echo)
            reply = out.split("Reply in the user's style, concise")[-1].strip()
            if not reply:
                reply = out.strip()
            if len(reply) > 280:
                reply = reply[:277] + "..."
            return reply
        else:
            logger.warning("HF status %s: %s", r.status_code, r.text[:200])
    except Exception as e:
        logger.error("HF generation error: %s", e)
    # fallback short reply
    return f"Interesting — curious how this will affect startups."

# --- Fetch user style examples (recent tweets) ---
def fetch_user_style_examples(max_tweets=15):
    try:
        resp = client_v2.get_users_tweets(id=YOUR_USER_ID, max_results=max_tweets, exclude=['retweets'])
        if resp.data:
            texts = [t.text for t in resp.data]
            return " ||| ".join(texts[:10])
    except Exception as e:
        logger.warning("Could not fetch user tweets: %s", e)
    return "Concise, insightful takes on tech and startups."

# --- Get global trends and filter for niches (uses v1.1 trends/place id=1 worldwide) ---
def get_relevant_trends():
    try:
        places = api_v1.get_place_trends(id=1)  # worldwide
        if places and len(places) > 0:
            trends = [t["name"] for t in places[0]["trends"]]
            relevant = [t for t in trends if any(k.lower() in t.lower() for k in NICHE_KEYWORDS)]
            logger.info("Found trends: %s", relevant[:8])
            return relevant[:8]
    except Exception as e:
        logger.warning("Trends error: %s", e)
    # fallback: use niche keywords themselves
    return NICHE_KEYWORDS[:5]

# --- Find target tweets for a trend (v2 search) ---
def find_target_tweets(trend, max_results=5):
    query = f'"{trend}" lang:en -is:retweet'
    try:
        tweets = client_v2.search_recent_tweets(query=query, max_results=max_results, tweet_fields=['id','text','author_id','public_metrics','created_at'])
        if tweets and tweets.data:
            recent_threshold = datetime.utcnow() - timedelta(hours=3)
            results = []
            for t in tweets.data:
                # skip own tweets
                if str(t.author_id) == str(YOUR_USER_ID):
                    continue
                # basic popularity filter
                likes = t.public_metrics.get("like_count", 0)
                if likes < 5:
                    continue
                # add
                results.append(t)
            logger.info("Targets for %s: %d", trend, len(results))
            return results[:3]
    except Exception as e:
        logger.warning("Search error for %s: %s", trend, e)
    return []

# --- Safety: filter out extreme political/unsafe content if slider low ---
def safe_to_reply(text):
    if CONTROVERSY_SLIDER <= 2:
        bad_terms = ["kill","terror","bomb","rape","child"]  # very small filter; expand as needed
        if any(bt in text.lower() for bt in bad_terms):
            return False
    # prevent replies to obvious controversy triggers
    return True

# --- Post a reply ---
def post_reply(tweet_id, reply_text):
    try:
        resp = client_v2.create_tweet(text=reply_text, in_reply_to_tweet_id=tweet_id)
        if resp and getattr(resp, "data", None):
            reply_id = resp.data.get("id")
            logger.info("Posted reply id=%s", reply_id)
            return reply_id
    except Exception as e:
        logger.error("Post error: %s", e)
    return None

# --- Main run ---
def main():
    state = load_state()
    today = datetime.utcnow().date().isoformat()
    if state.get("date") != today:
        state["date"] = today
        state["replies_today"] = 0

    if state.get("replies_today", 0) >= MAX_REPLIES_PER_DAY:
        logger.info("Daily limit reached (%s/%s). Exiting.", state['replies_today'], MAX_REPLIES_PER_DAY)
        save_state(state)
        return

    style_examples = fetch_user_style_examples()
    trends = get_relevant_trends()
    random.shuffle(trends)

    posted = 0
    for trend in trends:
        if state["replies_today"] + posted >= MAX_REPLIES_PER_DAY:
            break
        targets = find_target_tweets(trend)
        for t in targets:
            if state["replies_today"] + posted >= MAX_REPLIES_PER_DAY:
                break
            tweet_text = t.text
            if not safe_to_reply(tweet_text):
                logger.info("Skipping unsafe tweet.")
                continue
            reply = generate_reply_hf(tweet_text, style_examples)
            # tiny randomization to avoid identical patterns
            reply = reply.strip()
            if not reply:
                continue
            # post
            reply_id = post_reply(t.id, reply)
            if reply_id:
                posted += 1
                # small sleep between posts
                time.sleep(random.randint(10, 30))
        # small delay between trend groups
        time.sleep(random.randint(5, 12))

    state["replies_today"] += posted
    logger.info("Session complete. Posted %s replies. Total today: %s", posted, state["replies_today"])
    save_state(state)

if __name__ == "__main__":
    main()
