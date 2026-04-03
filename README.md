# Customer Personality Analysis: Machine Learning for Marketing Campaigns

## Project Overview
This project investigates customer behavior and predicts responses to marketing campaigns for a retail client. The primary goal is to provide an efficient method to prioritize customers for future campaigns and customize marketing messages based on customer segments.

### Business Questions Addressed:
1.  What customer segments exist based on value, recency, purchase frequency, and channel behavior?
2.  What are the strongest correlations to campaign response?
3.  Who are the best customers to target, based on their predicted likelihood of responding?

## Dataset
The project uses the "Customer Personality Analysis" dataset from Kaggle.com, comprising 2,240 customer records and 29 variables, including:
*   **Demographics:** Age, Income, Education, Marital Status
*   **Spending Behavior:** Amount spent on various products (wines, meat, fruits, etc.)
*   **Customer Activity:** Number of purchases by channel, recency (days since last purchase)
*   **Marketing Response:** Binary variables for acceptance of previous campaigns and the most recent campaign.

## Methodology

### 1. Data Cleaning & Feature Engineering
*   Handled missing `Income` values by imputation with the mean.
*   Created an `Age` feature from `Year_Birth` and capped unrealistic ages.
*   Dropped identifier (`ID`) and constant columns (`Z_CostContact`, `Z_Revenue`).
*   Addressed negative values in spending and purchase-related columns.
*   Clipped extreme outliers in `Income` and spending features.
*   Engineered new features: `Total_Spending`, `Total_Purchases`, and `Spending_per_Purchase` to capture overall customer value and engagement.

### 2. Exploratory Data Analysis (EDA)
*   **Boxplot Analysis:** Visualized feature distributions, confirming the need for scaling due to varying scales and outliers.
*   **Income Distribution:** Showed a slight right-skew, justifying log-transformation.
*   **Spending Correlation Heatmap:** Identified strong positive correlations between spending categories, particularly wine and meat, indicating high-value customers spend across multiple areas.
*   **Target Variable Distribution:** Revealed a class imbalance (14.9% response rate), highlighting the need for appropriate evaluation metrics.
*   **Feature–Response Correlation (Point-Biserial):** Identified key features like `Total_Spending`, `MntMeatProducts`, `NumWebPurchases`, and `Recency` as strong predictors of campaign response.
*   **Log-Odds Linearity Check:** Confirmed near-linear trends for several features, supporting Logistic Regression.
*   **Non-Linear Patterns and Interaction Effects:** Illustrated interactions between Age, Income, Recency, and Web Purchases, suggesting the value of Decision Trees.

### 3. Clustering (K-Means)
*   **Objective:** Segment customers into distinct behavioral groups.
*   **Features:** `Income` (log-transformed and outlier-treated), `Age`, `Total_Spending`, `Total_Purchases`, `Spending_per_Purchase`.
*   **Preprocessing:** Outlier handling, log transformation of `Income`, and `StandardScaler` for normalization.
*   **Dimensionality Reduction:** PCA was applied to reduce feature redundancy and facilitate 2D visualization.
*   **Optimal Clusters:** The Elbow Method, Silhouette Score, and Davies–Bouldin Index consistently suggested 3 optimal clusters.
*   **Cluster Profiles:** Identified three distinct segments:
    *   **Cluster 0:** High-income, high-spending, high-frequency purchasers.
    *   **Cluster 1:** Lower-income, low-spending, low-engagement customers.
    *   **Cluster 2:** Middle-income, moderate-spending, and oldest on average.

### 4. Predictive Modeling
**Objective:** Predict customer response to marketing campaigns.

#### a. Logistic Regression
*   **Features:** All relevant demographic, spending, and purchase behavior features, one-hot encoded for categorical variables.
*   **Preprocessing:** `StandardScaler` applied to numerical features.
*   **Class Imbalance:** Addressed using SMOTE during training.
*   **Interpretation:** Coefficients and odds ratios provided insights into feature importance:
    *   **Positive Indicators:** Prior campaign acceptance, web visits, meat spending, higher education.
    *   **Negative Indicators:** High recency (inactive), presence of teenagers, married/cohabiting status, in-store purchases.

#### b. Decision Tree
*   **Features:** Similar to Logistic Regression, handling mixed data types intrinsically.
*   **Training & Tuning:** Used `GridSearchCV` with `average_precision` scoring to optimize hyperparameters, including `class_weight` to manage imbalance.
*   **Threshold Tuning:** Optimized the classification threshold on a validation set to maximize F1-score and Recall, selecting 0.7.
*   **Interpretation:** The tree structure and feature importances revealed decision rules for response prediction. Key features included `Total_Spending`, `NumWebPurchases`, `Recency`, `MntMeatProducts`, and `Income`.

### 5. Model Comparison
*   **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices were used.
*   **Performance:** 
    *   **Logistic Regression:** Achieved an ROC-AUC of 0.827, showing slightly better overall discrimination.
    *   **Decision Tree:** Achieved an ROC-AUC of 0.802. Crucially, it provided a **lift of 2.97** in response rate (44.4% vs 15.0% overall) when targeting the top 20% of customers.

## Recommendations

1.  **Decision Tree for Targeting:** Utilize the Decision Tree for prioritizing and targeting customers due to its high lift and interpretable if-then rules, which are easy to implement in CRM systems. This will significantly improve campaign efficiency by focusing on the highest-probability responders.
2.  **Logistic Regression for Explanation:** Employ Logistic Regression for explaining and justifying marketing decisions. Its interpretable coefficients and stable probability scores offer valuable insights into *why* certain customer attributes influence response, supporting strategic planning.

The two models are complementary, with the Decision Tree driving operational targeting and Logistic Regression providing strategic understanding.

## Ethical, Privacy, and Security Considerations
*   **Ethical:** Guard against unfair targeting. Regularly review selected customer lists to ensure no groups are consistently excluded based on proxies for social advantage.
*   **Privacy:** Treat customer-level scores as sensitive data. Use minimal fields, anonymize identifiers where possible, and restrict access to campaign teams only.
*   **Security:** Implement strict access controls for customer score lists. Ensure secure storage, audit trails, and limited reuse of scores to mitigate data breach risks.
