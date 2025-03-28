# FusionTech Product Review Data Processing

## Overview
This document outlines the steps taken to process and clean Amazon product review data, resulting in two output files: a product summary and a cleaned review dataset.

## Data Cleaning and Validation

### 1. Check for Empty Values
- Identified products with empty price fields
- Counted items with missing OS information
- Tallied products lacking color data

### 2. Product Variation Analysis
- Detected ASINs with multiple color options
- Identified ASINs associated with various operating systems

### 3. Data Consistency Verification
- Checked for ASINs with inconsistent product titles
- Identified ASINs with varying feature descriptions

## Data Processing and Aggregation

### 1. Metric Calculations
- Computed average rating per ASIN
- Tallied total review count for each product
- Calculated mean price per ASIN

### 2. ASIN Summary Table Creation
- Merged average ratings, review counts, and prices
- Incorporated product attributes (title, features, OS, color)
- Used first occurrence for consistent attribute representation

### 3. Original Dataset Cleaning
- Removed redundant and unnecessary columns:
  - brand, user_id, main_category, store, categories
  - bought_together, subtitle, author, num_reviews
  - average_rating, rating_number, avg_helpful_votes

## Output Generation

### 1. ASIN Summary File
- Filename: `asin_summary.csv`
- Contents: Aggregated product-level data
- Columns: ASIN, avg_rating, num_reviews, price, title, features, OS, color

### 2. Cleaned Reviews File
- Filename: `cleaned_reviews.csv`
- Contents: Original review data with irrelevant columns removed
- Retained essential review information

## Execution Summary
- Processed [X] unique products
- Cleaned [Y] individual review entries
- Generated two output CSV files for further analysis

---
Answer from Perplexity: pplx.ai/share