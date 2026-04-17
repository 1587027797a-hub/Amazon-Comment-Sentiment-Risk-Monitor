# Sentiment Polarization Index Calculation | ECON7055 Research Project
# Step 3: Calculate Polarization Metrics by Topic/Year/Risk


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 70)
print("📊 Sentiment Polarization Index Calculation | ECON7055 Research Project")
print("=" * 70)

#Configuration
# 修改：匹配 Part 2 的输出文件名
INPUT_FILE = 'All_Beauty_disputes_complete_EN_2021_2023_with_sentiment.csv'
OUTPUT_FILE = 'All_Beauty_polarization_analysis_complete.csv'
OUTPUT_FIG = 'All_Beauty_polarization_visualization_complete.png'

print(f"\n📂 Input: {INPUT_FILE}")

# Check file existence
if not os.path.exists(INPUT_FILE):
    print(f"\n❌ Error: Input file '{INPUT_FILE}' not found!")
    print("   Please run Part 2 (Sentiment Analysis) first.")
    exit()

#  Load Data
print("\n📂 Loading data with sentiment scores...")
df = pd.read_csv(INPUT_FILE, encoding='utf-8')
print(f"✅ Total reviews: {len(df)}")

# Critical Column Checks
required_cols = ['sentiment_score', 'cleaned_text']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"\n❌ Error: Missing required columns: {missing_cols}")
    exit()

print("   • Verified: sentiment_score, cleaned_text")
if 'risk_label' in df.columns:
    print("   • Verified: risk_label (Ready for Risk Analysis)")
if 'year' in df.columns:
    print("   • Verified: year (Ready for Time Analysis)")


# Polarization Index Formula
def calculate_polarization_index(scores):
    """
    Calculate multi-dimensional polarization index
    Components:
    1. Sentiment Mean: overall tendency (higher = more negative)
    2. Sentiment Variance: opinion dispersion (higher = more polarized)
    3. Extreme Ratio: proportion of extreme negative (score > 0.8)
    Final Index = 0.3*Mean + 0.4*Variance + 0.3*Extreme_Ratio
    """
    if len(scores) == 0:
        return {
            'mean': np.nan,
            'variance': np.nan,
            'extreme_ratio': np.nan,
            'polarization_index': np.nan,
            'count': 0
        }

    scores = np.array(scores)
    # Handle potential NaNs in scores
    scores = scores[~np.isnan(scores)]

    if len(scores) == 0:
        return {
            'mean': np.nan, 'variance': np.nan,
            'extreme_ratio': np.nan, 'polarization_index': np.nan, 'count': 0
        }

    mean_sentiment = np.mean(scores)
    var_sentiment = np.var(scores)
    extreme_ratio = np.sum(scores > 0.8) / len(scores)

    # Normalized polarization index (0-1)
    # Note: Variance is typically small (0-0.25), so weight 0.4 is appropriate
    polarization_index = 0.3 * mean_sentiment + 0.4 * var_sentiment + 0.3 * extreme_ratio

    return {
        'mean': mean_sentiment,
        'variance': var_sentiment,
        'extreme_ratio': extreme_ratio,
        'polarization_index': polarization_index,
        'count': len(scores)
    }


# Overall Polarization
print("\n📈 Overall Polarization Analysis:")
overall = calculate_polarization_index(df['sentiment_score'].tolist())
print(f"   • Mean Sentiment: {overall['mean']:.4f}")
print(f"   • Variance: {overall['variance']:.4f}")
print(f"   • Extreme Ratio: {overall['extreme_ratio']:.4f}")
print(f"   • Polarization Index: {overall['polarization_index']:.4f}")
print(f"   • Total Reviews: {overall['count']}")

#  Polarization by Year
print("\n📅 Polarization by Year:")
yearly_results = []
if 'year' in df.columns:
    for year in sorted(df['year'].unique()):
        # Handle NaN years
        year_data = df[df['year'] == year]['sentiment_score'].dropna().tolist()
        if len(year_data) > 0:
            metrics = calculate_polarization_index(year_data)
            metrics['year'] = year
            yearly_results.append(metrics)
            print(f"   • {year}: PI={metrics['polarization_index']:.4f} (n={metrics['count']})")
    df_yearly = pd.DataFrame(yearly_results)
else:
    df_yearly = None
    print("   ⚠️ No year column found")

# Polarization by Risk Label (NEW & CRITICAL)
print("\n🏷️ Polarization by Risk Label (High/Medium/Low):")
risk_results = []
if 'risk_label' in df.columns:
    # Ensure order: High, Medium, Low
    risk_order = ['High', 'Medium', 'Low']
    for label in risk_order:
        label_data = df[df['risk_label'] == label]['sentiment_score'].dropna().tolist()
        if len(label_data) > 0:
            metrics = calculate_polarization_index(label_data)
            metrics['risk_label'] = label
            risk_results.append(metrics)
            print(f"   • {label} Risk: PI={metrics['polarization_index']:.4f} (n={metrics['count']})")
    df_risk = pd.DataFrame(risk_results)
else:
    df_risk = None
    print("   ⚠️ No risk_label column found")

# Polarization by Dispute Keyword
print("\n🔑 Polarization by Dispute Keyword:")
keyword_results = []
keywords_to_check = ['refund', 'return', 'logistics', 'customs', 'fraud', 'delay', 'scam', 'damaged', 'wrong']
for keyword in keywords_to_check:
    keyword_data = df[df['cleaned_text'].str.contains(keyword, na=False)]['sentiment_score'].dropna().tolist()
    if len(keyword_data) > 0:
        metrics = calculate_polarization_index(keyword_data)
        metrics['keyword'] = keyword
        keyword_results.append(metrics)
        print(f"   • {keyword}: PI={metrics['polarization_index']:.4f} (n={len(keyword_data)})")
df_keyword = pd.DataFrame(keyword_results)

# Polarization by Rating
print("\n⭐ Polarization by Rating:")
rating_results = []
if 'rating' in df.columns:
    for rating in sorted(df['rating'].unique()):
        rating_data = df[df['rating'] == rating]['sentiment_score'].dropna().tolist()
        if len(rating_data) > 0:
            metrics = calculate_polarization_index(rating_data)
            metrics['rating'] = rating
            rating_results.append(metrics)
            print(f"   • {rating} stars: PI={metrics['polarization_index']:.4f} (n={metrics['count']})")
    df_rating = pd.DataFrame(rating_results)
else:
    df_rating = None

# Extreme Negative Reviews Analysis
print("\n🚨 Extreme Negative Reviews Analysis (Score > 0.9):")

# Filter extreme negative reviews
extreme_df = df[df['sentiment_score'] > 0.9].copy()
n_extreme = len(extreme_df)
total = len(df)
extreme_pct = n_extreme / total * 100

print(f"   • Total extreme negative reviews: {n_extreme} ({extreme_pct:.1f}% of total)")

if n_extreme > 0:
    print(f"   • Average sentiment score: {extreme_df['sentiment_score'].mean():.4f}")

    # Top keywords in extreme reviews
    print(f"\n   🔍 Top dispute keywords in extreme reviews:")
    extreme_keywords_freq = {}
    for keyword in keywords_to_check:
        count = extreme_df['cleaned_text'].str.contains(keyword, na=False).sum()
        if count > 0:
            extreme_keywords_freq[keyword] = count
            print(f"      • {keyword}: {count} ({count / n_extreme * 100:.1f}%)")

    # Save extreme reviews
    extreme_output_file = '.venv/All_Beauty_extreme_negative_reviews_complete.csv'
    extreme_df.to_csv(extreme_output_file, index=False, encoding='utf-8')
    print(f"\n   ✅ Saved extreme reviews to: {extreme_output_file}")
else:
    print("   ⚠️ No extreme negative reviews found (score > 0.9)")

# Save Analysis Results
print(f"\n💾 Saving analysis results...")

all_rows = []

# 1. Overall
all_rows.append({
    'dimension': 'Overall', 'category': 'All',
    'mean_sentiment': overall['mean'], 'variance': overall['variance'],
    'extreme_ratio': overall['extreme_ratio'], 'polarization_index': overall['polarization_index'],
    'count': overall['count']
})

# 2. Year
if df_yearly is not None:
    for _, row in df_yearly.iterrows():
        all_rows.append({
            'dimension': 'Year', 'category': str(row['year']),
            'mean_sentiment': row['mean'], 'variance': row['variance'],
            'extreme_ratio': row['extreme_ratio'], 'polarization_index': row['polarization_index'],
            'count': row['count']
        })

# 3. Risk Label (New)
if df_risk is not None:
    for _, row in df_risk.iterrows():
        all_rows.append({
            'dimension': 'Risk_Label', 'category': row['risk_label'],
            'mean_sentiment': row['mean'], 'variance': row['variance'],
            'extreme_ratio': row['extreme_ratio'], 'polarization_index': row['polarization_index'],
            'count': row['count']
        })

# 4. Keyword
if len(df_keyword) > 0:
    for _, row in df_keyword.iterrows():
        all_rows.append({
            'dimension': 'Keyword', 'category': row['keyword'],
            'mean_sentiment': row['mean'], 'variance': row['variance'],
            'extreme_ratio': row['extreme_ratio'], 'polarization_index': row['polarization_index'],
            'count': row['count']
        })

# 5. Rating
if df_rating is not None:
    for _, row in df_rating.iterrows():
        all_rows.append({
            'dimension': 'Rating', 'category': str(row['rating']),
            'mean_sentiment': row['mean'], 'variance': row['variance'],
            'extreme_ratio': row['extreme_ratio'], 'polarization_index': row['polarization_index'],
            'count': row['count']
        })

df_summary = pd.DataFrame(all_rows)
df_summary.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
print(f"✅ Saved: {OUTPUT_FILE}")

#Generate Visualization
print("\n📊 Generating visualization...")

try:
    # Font handling for compatibility
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Sentiment Polarization Analysis (2021-2023)', fontsize=16, fontweight='bold')

    # 1. Polarization Index by Keyword
    ax1 = plt.subplot(3, 2, 1)
    if len(df_keyword) > 0:
        df_keyword_sorted = df_keyword.sort_values('polarization_index', ascending=True)
        ax1.barh(df_keyword_sorted['keyword'], df_keyword_sorted['polarization_index'],
                 color='steelblue', edgecolor='black')
        ax1.set_title('Polarization Index by Dispute Keyword', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Polarization Index (0-1)')
        ax1.grid(axis='x', alpha=0.3)

    # 2. Polarization Trend by Year
    ax2 = plt.subplot(3, 2, 2)
    if df_yearly is not None and len(df_yearly) > 0:
        ax2.plot(df_yearly['year'], df_yearly['polarization_index'],
                 marker='o', linewidth=2, markersize=8, color='steelblue')
        ax2.fill_between(df_yearly['year'], df_yearly['polarization_index'], alpha=0.3)
        ax2.set_title('Polarization Index Trend by Year', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Polarization Index')
        ax2.grid(alpha=0.3)

    # 3. Sentiment Score Distribution
    ax3 = plt.subplot(3, 2, 3)
    ax3.hist(df['sentiment_score'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(x=0.6, color='red', linestyle='--', linewidth=2, label='Negative Threshold (0.6)')
    ax3.axvline(x=0.9, color='orange', linestyle='--', linewidth=2, label='Extreme Threshold (0.9)')
    ax3.set_title('Sentiment Score Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Sentiment Score (0-1, higher=negative)')
    ax3.set_ylabel('Frequency')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # 4. Polarization by Risk Label (NEW & IMPORTANT)
    ax4 = plt.subplot(3, 2, 4)
    if df_risk is not None and len(df_risk) > 0:
        # Ensure order High, Medium, Low
        df_risk_sorted = df_risk.set_index('risk_label').reindex(['High', 'Medium', 'Low']).reset_index()
        ax4.bar(df_risk_sorted['risk_label'], df_risk_sorted['polarization_index'],
                color=['darkred', 'orange', 'green'], edgecolor='black', alpha=0.8)
        ax4.set_title('Polarization Index by Risk Label', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Risk Level')
        ax4.set_ylabel('Polarization Index')
        ax4.grid(axis='y', alpha=0.3)

        for i, v in enumerate(df_risk_sorted['polarization_index']):
            if not np.isnan(v):
                ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

    # 5. Extreme Negative Reviews Distribution by Year
    ax5 = plt.subplot(3, 2, 5)
    if 'year' in df.columns and n_extreme > 0:
        extreme_by_year = extreme_df['year'].value_counts().sort_index()
        all_by_year = df['year'].value_counts().sort_index()
        extreme_pct_by_year = (extreme_by_year / all_by_year * 100).fillna(0)

        ax5.bar(extreme_pct_by_year.index.astype(str), extreme_pct_by_year.values,
                color='coral', edgecolor='black', alpha=0.8)
        ax5.set_title('Extreme Negative Reviews (% of Total) by Year', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Year')
        ax5.set_ylabel('Percentage (%)')
        ax5.grid(axis='y', alpha=0.3)

    # 6. Top Keywords in Extreme Reviews
    ax6 = plt.subplot(3, 2, 6)
    if n_extreme > 0 and extreme_keywords_freq:
        extreme_kw_df = pd.DataFrame(list(extreme_keywords_freq.items()),
                                     columns=['keyword', 'count'])
        extreme_kw_df['percentage'] = extreme_kw_df['count'] / n_extreme * 100
        extreme_kw_df = extreme_kw_df.sort_values('percentage', ascending=True)

        ax6.barh(extreme_kw_df['keyword'], extreme_kw_df['percentage'],
                 color='darkred', edgecolor='black', alpha=0.8)
        ax6.set_title('Keywords in Extreme Reviews (Score > 0.9)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Percentage of Extreme Reviews (%)')
        ax6.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved visualization: {OUTPUT_FIG}")
    plt.close()

except Exception as e:
    print(f"   ⚠️ Visualization failed: {e}")
    print("      Try: pip install matplotlib")

print("\n" + "=" * 70)
print("✅ Step 3 Complete! Ready for Model Training (Part 4)")
print("=" * 70)