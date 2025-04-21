# 🌍 Predicting CO₂ Emissions Using Machine Learning

This project explores how machine learning can be used to predict CO₂ emissions using historical data from 1975 to 2021. The work was developed as part of my final assignment for the Machine Learning course at The Hague University of Applied Sciences.

## 📊 Project Overview

- **Language**: R  
- **Techniques**: Supervised learning (regression), unsupervised learning (clustering)  
- **Models used**:
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Support Vector Regression (SVR)
  - Random Forest
  - K-Means Clustering

## 🧪 Workflow

- Data cleaning & handling missing values (with median imputation)
- Z-score normalization for numeric features
- Exploratory data analysis and visualizations
- Train-test split (80/20)
- Model training and evaluation using RMSE and R²
- Feature importance visualization (Random Forest)
- Cluster analysis for pattern discovery

## 📂 Repository Contents

| File | Description |
|------|-------------|
| `ML_Paper_2_with_Citations.pdf` | Final research paper |
| `co2_emissions_analysis.R` | Full R code with comments |
| `sample_display_data.csv` | Sample of cleaned dataset (10 rows) |

## 📸 Key Visuals

- Histogram of CO₂ emissions (scaled)
- Time series trends by country and average
- Correlation heatmap across years
- Feature importance from Random Forest
- K-Means cluster visualization

## 💡 Insights

- SVR achieved the highest accuracy (lowest RMSE, highest R²)
- Random Forest helped interpret feature importance (e.g., 2015, 2019)
- Clustering revealed similarities between countries’ emission patterns

## 🧠 Skills Demonstrated

- Data preprocessing and wrangling  
- Supervised/unsupervised modeling  
- Model comparison and evaluation  
- Data visualization in `ggplot2`, `corrplot`, and `factoextra`  
- Reproducible research workflow in R

---

Feel free to ⭐ the repo or reach out with any questions!

