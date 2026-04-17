# Reddit Data Pipeline | ECON7055 Research Project
# Step 1: Data Cleaning & Risk Label Definition (No Rating)

import pandas as pd
import numpy as np
import re
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 70)
print("🔄 Reddit Data Pipeline | ECON7055 Research Project")
print("=" * 70)

# Input
INPUT_FILE = 'Reddit_ecommerce_disputes_posts.csv'
OUTPUT_FILE = 'Reddit_disputes_complete_EN_2021_2023.csv'
STATS_FILE = 'Reddit_disputes_stats_EN_2021_2023.txt'

# Controversial keywords (consistent with Amazon)
DISPUTE_KEYWORDS = [
    'refund', 'return', 'logistics', 'customs',
    'fraud', 'delay', 'scam', 'damaged', 'wrong'
]

# Time Range
START_YEAR = 2021
END_YEAR = 2023

# Built-in stop word
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
    'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
    'now', 'would', 'could', 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
    'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

print(f"\n📂 Input: {INPUT_FILE}")
print(f"🔑 Dispute Keywords: {DISPUTE_KEYWORDS}")
print(f"⏰ Time Range: {START_YEAR}-{END_YEAR}")

#1.data loading
print("\n[1/7] data loading...")

if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: File not found '{INPUT_FILE}'")
    exit()

df = pd.read_csv(INPUT_FILE, encoding='utf-8', on_bad_lines='skip')
print(f"   ✅ Loaded {len(df)} posts")

# Check the necessary columns
required = ['post_id', 'title', 'selftext']
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"❌ Error: Missing columns {missing}")
    print("   Current column:", list(df.columns))
    exit()

# 2. Basic Cleaning
print("\n[2/7] Basic Cleaning...")

# de-weight
df_before = len(df)
df = df.drop_duplicates(subset=['post_id'])
print(f"   •Remove duplicates: {df_before - len(df)}")

# Merge text columns (title + selftext)
df['raw_text'] = (df['title'].fillna('') + ' ' + df['selftext'].fillna('')).str.strip()

# Remove the empty text
df_before = len(df)
df = df[
    (df['raw_text'] != '') &
    (~df['raw_text'].str.lower().isin(['[removed]', '[deleted]', '']))
    ]
print(f"   • Remove empty/delete content: {df_before - len(df)}")

# 3. Text cleaning
print("\n[3/7] Text cleaning...")


def clean_text(text):
    """Unified text cleaning function"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Remove URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove Reddit label
    text = re.sub(r'\[removed\]|\[deleted\]', '', text, flags=re.IGNORECASE)
    # Only keep the letters and Spaces
    text = re.sub(r'[^A-Za-z\s]', ' ', text)
    # Normalized lowercase Spaces
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


df['cleaned_text'] = df['raw_text'].apply(clean_text)

# Remove what is empty after cleaning
df_before = len(df)
df = df[df['cleaned_text'] != '']
print(f"   • Effective after cleaning: {len(df)} (Remove {df_before - len(df)})")

# 4. Keyword screening
print("\n[4/7] Keyword screening...")


def find_keywords(text, keywords):
    if pd.isna(text):
        return []
    return [kw for kw in keywords if kw in text.lower()]


df['dispute_keywords'] = df['cleaned_text'].apply(
    lambda x: find_keywords(x, DISPUTE_KEYWORDS)
)
df['keyword_count'] = df['dispute_keywords'].apply(len)

# Screen posts that contain at least one keyword
df_before = len(df)
df = df[df['keyword_count'] > 0]
print(f"   • Contains controversial keywords: {len(df)} (from {df_before})")

# Keyword frequency statistics
from collections import Counter

all_kws = []
for kws in df['dispute_keywords']:
    all_kws.extend(kws)
keyword_freq = Counter(all_kws)

print("   🔑 Top keywords:")
for kw, cnt in keyword_freq.most_common(5):
    print(f"      • {kw}: {cnt}")

# 5. Time screening
print("\n[5/7] Time screening...")


def convert_ts(ts):
    try:
        if pd.isna(ts):
            return None
        return pd.to_datetime(ts, unit='s', errors='coerce')
    except:
        return None


# The time column on Reddit is usually created utc
time_col = 'created_utc' if 'created_utc' in df.columns else None

if time_col:
    df['post_date'] = df[time_col].apply(convert_ts)
    df['year'] = df['post_date'].dt.year
    df['month'] = df['post_date'].dt.month

    before = len(df)
    df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)]
    print(f"   • After time screening: {len(df)} (Remove {before - len(df)})")
else:
    print("   ⚠️ No time column, skip time filtering")
    df['year'] = None
    df['month'] = None

# 6. Participle + stop word
print("\n[6/7] Remove participle + stop word...")


def tokenize(text):
    if pd.isna(text) or text == '':
        return []
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2 and t.isalpha()]


df['tokens'] = df['cleaned_text'].apply(tokenize)
df = df[df['tokens'].apply(len) > 0]
df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))

print(f"   ✅ After word segmentation: {len(df)} ")

#7. Define temporary risk labels
print("\n[7/7] Define temporary risk labels...")

# Part 2 The sentiment analysis will be updated based on the sentiment score

def define_temp_risk(row):
    kw_count = row.get('keyword_count', 0)
    if kw_count >= 2:
        return 'High'
    elif kw_count == 1:
        return 'Medium'
    else:
        return 'Low'


df['risk_label'] = df.apply(define_temp_risk, axis=1)

print("   📊 Temporary risk distribution (based on keywords):")
for label, count in df['risk_label'].value_counts().items():
    pct = count / len(df) * 100
    print(f"      {label}: {count} ({pct:.1f}%)")

print("\n   ⚠️ Attention:")
print("      • The current risk labels are only based on the number of keywords")
print("      • Part 2 After the sentiment analysis, the tags will be updated based on the sentiment score")

# Save Results
print("\n💾 Save Results...")

columns_to_save = [
    'post_id', 'title', 'selftext', 'cleaned_text', 'processed_text',
    'tokens', 'author', 'score', 'post_date', 'year', 'month',
    'subreddit', 'url', 'dispute_keywords', 'keyword_count', 'risk_label'
]
available = [c for c in columns_to_save if c in df.columns]
df[available].to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
print(f"   ✅ Saved: {OUTPUT_FILE}")

# Save Statistics
with open(STATS_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write(f"Reddit Dispute Posts Stats ({START_YEAR}-{END_YEAR})\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Total posts: {len(df)}\n")
    if 'score' in df.columns:
        f.write(f"Score range: {df['score'].min()} - {df['score'].max()}\n")
        f.write(f"Average score: {df['score'].mean():.2f}\n")
    if 'post_date' in df.columns and df['post_date'].notna().any():
        f.write(f"Time range: {df['post_date'].min()} to {df['post_date'].max()}\n")
    f.write(f"\nRisk Label Distribution (temporary):\n")
    for label, count in df['risk_label'].value_counts().items():
        f.write(f"  {label}: {count}\n")
    f.write(f"\nKeyword frequency:\n")
    for kw, cnt in keyword_freq.most_common():
        f.write(f"  {kw}: {cnt}\n")
print(f"   ✅ Saved: {STATS_FILE}")

# Completed
print("\n" + "=" * 70)
print("✅ Reddit Data Pipeline Complete!")
print("=" * 70)

print(f"\n📁 Output File:")
print(f"   1. {OUTPUT_FILE}")
print(f"   2. {STATS_FILE}")

print(f"\n📊 Data summary:")
print(f"   • Total posts: {len(df)}")
print(f"   • Time Range: {START_YEAR}-{END_YEAR}")
print(f"   • Risk tag: Temporary (based on keywords, updated in Part 2)")

print("\n🎯 Next Step: Run reddit_sentiment_analysis.py")
print("   (Update risk labels through sentiment analysis")
print("=" * 70)