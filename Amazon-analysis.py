# RoBERTa Sentiment Analysis | ECON7055 Research Project
# Step 2: Sentiment Scoring with Pre-trained RoBERTa

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 70)
print("🤖 RoBERTa Sentiment Analysis | ECON7055 Research Project")
print("=" * 70)

#  Configuration
INPUT_FILE = 'All_Beauty_disputes_complete_EN_2021_2023.csv'
OUTPUT_FILE = 'All_Beauty_disputes_complete_EN_2021_2023_with_sentiment.csv'

# Use pre-trained sentiment model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 32
MAX_LENGTH = 128

print(f"\n📂 Input: {INPUT_FILE}")
print(f"🤖 Model: {MODEL_NAME}")
print(f"📦 Batch Size: {BATCH_SIZE}")

# Check file existence
if not os.path.exists(INPUT_FILE):
    print(f"\n❌ Error: Input file '{INPUT_FILE}' not found!")
    print("   Please run Part 1 (Data Pipeline) first.")
    exit()

# Load Model
print("\n⏳ Loading RoBERTa model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    exit()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Model loaded on: {device}")

# Load Data
print("\n📂 Loading cleaned data...")
df = pd.read_csv(INPUT_FILE, encoding='utf-8')
print(f"✅ Total reviews: {len(df)}")

# Check if risk_label exists (from Part 1)
if 'risk_label' in df.columns:
    print("   • Found 'risk_label' column (Preserved from Part 1)")
else:
    print("   ⚠️ Warning: 'risk_label' column not found. Evaluation might fail.")


# Sentiment Prediction Function
def predict_sentiment_batch(texts, model, tokenizer, batch_size=32, device='cpu'):
    """
    Predict sentiment for a batch of texts.
    Model: cardiffnlp/twitter-roberta-base-sentiment-latest
    Labels: 0=Negative, 1=Neutral, 2=Positive
    Returns: probability of class 0 (Negative) -> Higher score = More Negative
    """
    all_scores = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]

        # Handle empty strings gracefully
        batch_texts = [t if isinstance(t, str) and t != '' else " " for t in batch_texts]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)

        # Extract Negative sentiment score (Label 0)
        # Higher score = Higher probability of being Negative
        negative_scores = probabilities[:, 0].cpu().numpy()
        all_scores.extend(negative_scores)

        # Progress
        if (i + batch_size) % 1000 == 0 or (i + batch_size) >= total:
            print(
                f"   🔄 Processed: {min(i + batch_size, total)}/{total} ({min(i + batch_size, total) / total * 100:.1f}%)")

    return all_scores


#  Run Sentiment Analysis
print("\n🔄 Running sentiment analysis...")
texts = df['cleaned_text'].fillna('').tolist()
sentiment_scores = predict_sentiment_batch(texts, model, tokenizer, BATCH_SIZE, device)

# Add to DataFrame
df['sentiment_score'] = sentiment_scores  # 0-1, higher = more negative
df['sentiment_label'] = df['sentiment_score'].apply(
    lambda x: 'Negative' if x > 0.6 else ('Neutral' if x > 0.33 else 'Positive')
)

print(f"\n✅ Sentiment analysis completed!")
print(f"📊 Score range: {df['sentiment_score'].min():.3f} - {df['sentiment_score'].max():.3f}")
print(f"📊 Mean score: {df['sentiment_score'].mean():.3f}")

#  Sentiment Distribution
print(f"\n📈 Sentiment Distribution:")
neg_count = (df['sentiment_score'] > 0.6).sum()
neu_count = ((df['sentiment_score'] > 0.33) & (df['sentiment_score'] <= 0.6)).sum()
pos_count = (df['sentiment_score'] <= 0.33).sum()
total = len(df)

print(f"   • Negative (score>0.6): {neg_count} ({neg_count / total * 100:.1f}%)")
print(f"   • Neutral (0.33-0.6): {neu_count} ({neu_count / total * 100:.1f}%)")
print(f"   • Positive (score<0.33): {pos_count} ({pos_count / total * 100:.1f}%)")

# Validation Check (Risk vs Sentiment)
# Quick check to ensure High Risk aligns with Negative Sentiment
if 'risk_label' in df.columns:
    print(f"\n🔍 Validation: Risk Label vs Sentiment Score")
    risk_sentiment = df.groupby('risk_label')['sentiment_score'].mean()
    for label, score in risk_sentiment.items():
        print(f"   • {label} Risk Avg Sentiment: {score:.3f} (Higher = More Negative)")
    # Expectation: High Risk should have higher sentiment score than Low Risk
else:
    print("\n⚠️ Skipping Risk Validation (risk_label missing)")

# Save Results 
print(f"\n💾 Saving results...")
# Ensure all columns including risk_label are saved
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
print(f"✅ Saved: {OUTPUT_FILE}")

# Clear CUDA cache if used
if device.type == 'cuda':
    torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("✅ Step 2 Complete! Ready for Model Training/Evaluation (Part 3)")
print("=" * 70)

