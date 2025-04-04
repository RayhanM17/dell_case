import pandas as pd

# Load data
df = pd.read_csv('../rawdata/reviews.csv', index_col='id')

# Data Cleaning Steps ---------------------------------------------------------

# 1. Check for empty price, OS, and color
def check_empty_values():
    empty_price = df[df['price'].isna()]
    empty_os = df[df['os'].isna()]
    empty_color = df[df['color'].isna()]
    
    print(f"Products with empty price: {len(empty_price)}")
    print(f"Products with empty OS: {len(empty_os)}")
    print(f"Products with empty color: {len(empty_color)}")

# 2. Check for multiple colors/OS per ASIN
def check_variations():
    color_counts = df.groupby('asin')['color'].nunique()
    os_counts = df.groupby('asin')['os'].nunique()
    
    multi_color = color_counts[color_counts > 1]
    multi_os = os_counts[os_counts > 1]
    
    print(f"\nASINs with multiple colors: {len(multi_color)}")
    print(f"ASINs with multiple OS: {len(multi_os)}")

# 3. Check title/features consistency
def check_consistency():
    title_counts = df.groupby('asin')['title_y'].nunique()
    feature_counts = df.groupby('asin')['features'].nunique()
    
    inconsistent_titles = title_counts[title_counts > 1]
    inconsistent_features = feature_counts[feature_counts > 1]
    
    print(f"\nASINs with inconsistent titles: {len(inconsistent_titles)}")
    print(f"ASINs with inconsistent features: {len(inconsistent_features)}")

# Run checks
print("Data Quality Checks:")
check_empty_values()
check_variations()
check_consistency()

# Data Processing -------------------------------------------------------------

# Calculate metrics
asin_avg_rating = df.groupby('asin')['rating'].mean().reset_index()
asin_avg_rating.rename(columns={'rating': 'avg_rating'}, inplace=True)

asin_review_count = df.groupby('asin').size().reset_index(name='num_reviews')
asin_avg_price = df.groupby('asin')['price'].mean().reset_index()  # Changed to mean()

# Create summary table
asin_summary = pd.merge(asin_avg_rating, asin_review_count, on='asin')
asin_summary = pd.merge(asin_summary, asin_avg_price, on='asin')

# Add product attributes (using first occurrence)
product_attributes = df.groupby('asin')[['title_y', 'features', 'os', 'color']].first().reset_index()
asin_summary = pd.merge(asin_summary, product_attributes, on='asin', how='left')

# Clean original dataset
columns_to_remove = [
    'brand', 'user_id', 'main_category', 'store', 'categories',
    'bought_together', 'subtitle', 'author', 'num_reviews',
    'average_rating', 'rating_number', 'avg_helpful_votes', 'os', 'color'
]
df_clean = df.drop(columns=columns_to_remove)

# Export files -----------------------------------------------------------------
asin_summary.to_csv('./rawdata/asin_summary.csv', index=False)
df_clean.to_csv('./rawdata/cleaned_reviews.csv', index=False)

print("\nProcessing complete:")
print(f"- ASIN summary saved to asin_summary.csv ({len(asin_summary)} products)")
print(f"- Cleaned reviews saved to cleaned_reviews.csv ({len(df_clean)} rows)")