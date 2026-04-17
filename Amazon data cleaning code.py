# ============================================================================
# Amazon Dispute Review Pipeline (End-to-End) | PART 1: Data Acquisition
# Input: All_Beauty.jsonl
# Output: Complete dispute reviews (2021-2023, ALL ratings, with dispute keywords)
# ============================================================================

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🔄 Amazon Dispute Review Pipeline | ECON7055 Research Project")
print("=" * 70)

# ========== Configuration ==========
INPUT_FILE = 'All_Beauty.jsonl'  # The original JSONL file
OUTPUT_CSV = 'All_Beauty_disputes_complete_EN_2021_2023.csv'
STATS_FILE = 'All_Beauty_disputes_stats_EN_2021_2023.txt'

# Controversial key words
DISPUTE_KEYWORDS = [
    'refund', 'return', 'logistics', 'customs',
    'fraud', 'delay', 'scam', 'damaged', 'wrong'
]

# Time range & scoring threshold
START_YEAR = 2021
END_YEAR = 2023
RATING_THRESHOLD = 3.0

print(f"\n⏰ Time Range: {START_YEAR} to {END_YEAR}")
print(f"⭐ Rating Strategy: Keep ALL ratings (1-5 stars) for Classification")
print(f"🔑 Dispute Keywords: {DISPUTE_KEYWORDS}")

# Step 1: Load Raw Data
print("\n📂 Step 1: Loading raw data from JSONL...")
df = pd.read_json(INPUT_FILE, lines=True)
print(f"✅ Loaded {len(df)} records")

# Step 2: Initial Filtering (Keywords ONLY, No Rating Filter)
print("\n🔍 Step 2: Initial filtering (Dispute Keywords ONLY)...")

# 2.1 Remove records without rating or text
df = df.dropna(subset=['rating', 'text'])
print(f"   • Removed missing rating/text: {len(df)} left")

# 2.2 Remove Rating Filter
df_filtered = df.copy()
print(f"   • Keeping ALL ratings (1-5 stars): {len(df_filtered)}")

# 2.3 Create search text (title + text)
df_filtered['search_text'] = (
    df_filtered['text'].fillna('') + ' ' + df_filtered['title'].fillna('')
).str.lower()

# 2.4 Filter by dispute keywords
pattern = '|'.join(DISPUTE_KEYWORDS)
mask = df_filtered['search_text'].str.contains(pattern, na=False)
df_disputes = df_filtered[mask].copy()
print(f"   • With dispute keywords: {len(df_disputes)}")

# Record which keywords were matched
def find_matched_keywords(text, keywords):
    if pd.isna(text):
        return []
    return [kw for kw in keywords if kw in text]

df_disputes['dispute_keywords'] = df_disputes['search_text'].apply(
    lambda x: find_matched_keywords(x, DISPUTE_KEYWORDS)
)

# Clean up temporary column
df_disputes = df_disputes.drop(columns=['search_text'])

# Step 3: Time Filtering
print(f"\n🕐 Step 3: Filtering by time ({START_YEAR}-{END_YEAR})...")

# Detect time column
time_col = 'timestamp' if 'timestamp' in df_disputes.columns else 'created_at' if 'created_at' in df_disputes.columns else None

if time_col:
    def convert_ts(ts):
        try:
            if pd.isna(ts):
                return None
            if isinstance(ts, (int, float)):
                return pd.to_datetime(ts, unit='s', errors='coerce')
            return pd.to_datetime(ts, errors='coerce')
        except:
            return None

    df_disputes['review_date'] = df_disputes[time_col].apply(convert_ts)
    df_disputes['year'] = df_disputes['review_date'].dt.year
    df_disputes['month'] = df_disputes['review_date'].dt.month

    # Filter year range
    before = len(df_disputes)
    df_disputes = df_disputes[
        (df_disputes['year'] >= START_YEAR) &
        (df_disputes['year'] <= END_YEAR)
    ]
    print(f"   • After time filter: {len(df_disputes)} (removed {before - len(df_disputes)})")
else:
    print("   ⚠️ No time column found. Skipping time filter.")

# Step 3.5: Create Risk Labels (Critical for Classification)
print("\n🏷️ Step 3.5: Creating Risk Labels based on Ratings...")

def assign_risk_label(rating):
    if rating < 3:
        return 'High'
    elif rating == 3:
        return 'Medium'
    else:
        return 'Low'

df_disputes['risk_label'] = df_disputes['rating'].apply(assign_risk_label)

# Show distribution to verify fix
risk_dist = df_disputes['risk_label'].value_counts()
print(f"   • Risk Distribution:")
for label, count in risk_dist.items():
    pct = (count / len(df_disputes)) * 100
    print(f"      {label}: {count} ({pct:.1f}%)")

# Step 4: Deep Text Cleaning
print("\n🧹 Step 4: Deep text cleaning...")

def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Remove HTML
    text = BeautifulSoup(text, 'html.parser').get_text()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Keep only letters and spaces
    text = re.sub(r'[^A-Za-z\s]', ' ', text)
    # Normalize whitespace and lowercase
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

df_disputes['cleaned_text'] = df_disputes['text'].apply(clean_text)

# Remove empty after cleaning
df_disputes = df_disputes[df_disputes['cleaned_text'] != '']
print(f"   • Non-empty after cleaning: {len(df_disputes)}")

# Step 5: Tokenization & Stop Words Removal
print("\n🔤 Step 5: Tokenization + stop words removal...")

STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
    'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
    'would', 'could', 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'wasn', "wasn't", 'weren', "weren't",
    'won', "won't", 'wouldn', "wouldn't"
}

def simple_tokenize(text):
    if pd.isna(text) or text == '' or not isinstance(text, str):
        return []
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2 and t.isalpha()]
    return tokens

df_disputes['tokens'] = df_disputes['cleaned_text'].apply(simple_tokenize)
df_disputes = df_disputes[df_disputes['tokens'].apply(len) > 0]
df_disputes['processed_text'] = df_disputes['tokens'].apply(lambda x: ' '.join(x))

print(f"   • Non-empty after tokenization: {len(df_disputes)}")

# Step 6: Final Stats & Save
print("\n✅ Step 6: Final statistics and saving...")

# Statistics
total = len(df_disputes)
avg_rating = df_disputes['rating'].mean() if 'rating' in df_disputes.columns else None
time_range = f"{df_disputes['review_date'].min()} to {df_disputes['review_date'].max()}" if 'review_date' in df_disputes.columns else "N/A"

print(f"\n📊 Final Dataset Summary:")
print(f"   • Total records: {total}")
print(f"   • Avg rating: {avg_rating:.2f}" if avg_rating else "   • Avg rating: N/A")
print(f"   • Time range: {time_range}")

# Keyword frequency
keyword_freq = {}
for kws in df_disputes['dispute_keywords']:
    for kw in kws:
        keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
print(f"\n📈 Top dispute keywords:")
for kw, cnt in sorted(keyword_freq.items(), key=lambda x: -x[1])[:5]:
    print(f"   • {kw}: {cnt}")

# Save
columns_to_save = [
    'rating', 'risk_label', 'title', 'text', 'cleaned_text', 'processed_text',
    'tokens', 'asin', 'user_id', 'review_date', 'year', 'month',
    'verified_purchase', 'helpful_vote', 'dispute_keywords'
]
available_cols = [col for col in columns_to_save if col in df_disputes.columns]
df_final = df_disputes[available_cols].copy()

df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"\n💾 Saved cleaned data to: {OUTPUT_CSV}")

# Save stats
with open(STATS_FILE, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write(f"Amazon Dispute Reviews Stats ({START_YEAR}-{END_YEAR})\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total records: {total}\n")
    if avg_rating:
        f.write(f"Average rating: {avg_rating:.2f}\n")
    f.write(f"Time range: {time_range}\n\n")
    f.write("Risk Label Distribution:\n")
    for label, count in risk_dist.items():
        f.write(f"  {label}: {count}\n")
    f.write("\nKeyword frequency:\n")
    for kw, cnt in sorted(keyword_freq.items(), key=lambda x: -x[1]):
        f.write(f"  {kw}: {cnt}\n")

print(f"📝 Saved stats to: {STATS_FILE}")
print("\n" + "="*70)
print("✅ End-to-end pipeline completed!")
print("="*70)