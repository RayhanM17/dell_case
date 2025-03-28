# Product Review Analysis Project

## Project Overview
This project focuses on analyzing and processing product review data to derive meaningful insights. It includes data cleaning, aggregation, and sentiment analysis, resulting in structured outputs for further exploration.

### Key Objectives:
1. Clean and validate raw review data.
2. Aggregate product-level metrics (e.g., average rating, price).
3. Perform sentiment analysis on customer reviews.
4. Analyze and Visualize review trends

---

## Directory Structure

### **rawdata**
Contains the raw and processed datasets:
- **`reviews.csv`**: Raw review data.
- **`asin_summary.csv`**: Aggregated product-level summary.
- **`cleaned_reviews.csv`**: Cleaned review dataset after removing irrelevant columns.

### **scripts**
Includes Python scripts and Jupyter notebooks for processing and analysis:
- **`datacleaning.ipynb`**: Jupyter notebook for cleaning and validating the dataset.
- **`datacleaning.py`**: Python script for automated data cleaning.
- **`analysis.ipynb`**: Notebook for exploring trends and performing advanced analysis.
- **`sentiment.ipynb`**: Notebook for performing sentiment analysis on customer reviews.

### **README.md**
Documentation file explaining the project structure, objectives, and steps taken.

### 1. Data Cleaning
Visit Data Cleaning markdown file for more information on data cleaning steps.

### 2. Sentiment Analysis
Performed sentiment analysis on customer reviews:
- Extracted sentiments (positive, neutral, negative) from review text.

### 3. Trend Analysis
exploring trends and performing advanced analysis.

---

## Outputs

### 1. Aggregated Product Summary (`asin_summary.csv`)
Contains product-level metrics:
- Columns: asin, avg_rating, num_reviews, avg_price, title_y, features, os, color.

### 2. Cleaned Reviews Dataset (`cleaned_reviews.csv`)
Contains cleaned review-level data:
- Columns: rating,title_x,text,asin,timestamp,helpful_vote,price

---

## How to Use

### **Data Cleaning**
Run the Python script or Jupyter notebook to clean the raw dataset:
```python
# python scripts/datacleaning.py
```

### **Data Analysis**
Use the `analysis.ipynb` notebook to explore trends such as:
- Rating distributions
- Price comparisons
- Review counts per product category

### **Sentiment Analysis**
Open the `sentiment.ipynb` notebook to analyze customer sentiments:
```python
# Example workflow:
# - Load cleaned_reviews.csv
# - Apply NLP techniques to extract sentiments
# - Visualize sentiment trends across products
```

---

## Requirements

### Python Libraries:
Ensure you have the following libraries installed:
```bash
pip install pandas numpy
```

---

## Future Work
1. Expand sentiment analysis with deep learning models.
2. Integrate additional datasets (e.g., competitor reviews).
3. Automate report generation with visualizations.

---