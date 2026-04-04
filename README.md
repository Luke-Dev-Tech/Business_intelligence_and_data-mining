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
  
Retrieve From (Kaggle): **https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis**

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


<p align="center">
  <img src="https://github.com/user-attachments/assets/cc351b16-372b-43fb-8931-a3ebc7e12123" 
       alt="Project Preview" 
       width="600"
       style="border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);" />
</p>

<p align="center">
  <em>Figure 1: EDA Boxplot Analysis of Min-Max</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/4bd9bc33-4415-4970-94f6-431ac78e063a" 
       alt="Project Preview" 
       width="600"
       style="border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);" />
</p>

<p align="center">
  <em>Figure 2: EDA Boxplot Analysis of Standard Deviation</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/3b5c1294-3484-4505-acce-847d0cf678df" 
       alt="Target Variable Distribution" 
       width="600" />
</p>

<p align="center">
  <em>Figure 3: Target Variable Distribution</em>
</p>

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

<p align="center">
  <img src="https://github.com/user-attachments/assets/0e88e6f5-ac13-49dd-81ec-0e98137edb1b" 
       alt="K-Means Clustering" 
       width="600" />
</p>

<p align="center">
  <em>Figure 4: K-Means Clustering</em>
</p>

### 4. Predictive Modeling
**Objective:** Predict customer response to marketing campaigns.

#### a. Logistic Regression
*   **Features:** All relevant demographic, spending, and purchase behavior features, one-hot encoded for categorical variables.
*   **Preprocessing:** `StandardScaler` applied to numerical features.
*   **Class Imbalance:** Addressed using SMOTE during training.
*   **Interpretation:** Coefficients and odds ratios provided insights into feature importance:
    *   **Positive Indicators:** Prior campaign acceptance, web visits, meat spending, higher education.
    *   **Negative Indicators:** High recency (inactive), presence of teenagers, married/cohabiting status, in-store purchases.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9dde8b28-b04e-4d9e-8c62-4f81ebd8a55c" 
       alt="Cofficient in Logistic Regression" 
       width="600" />
</p>

<p align="center">
  <em>Figure 5: Cofficient in Logistic Regression</em>
</p>

#### b. Decision Tree
*   **Features:** Similar to Logistic Regression, handling mixed data types intrinsically.
*   **Training & Tuning:** Used `GridSearchCV` with `average_precision` scoring to optimize hyperparameters, including `class_weight` to manage imbalance.
*   **Threshold Tuning:** Optimized the classification threshold on a validation set to maximize F1-score and Recall, selecting 0.7.
*   **Interpretation:** The tree structure and feature importances revealed decision rules for response prediction. Key features included `Total_Spending`, `NumWebPurchases`, `Recency`, `MntMeatProducts`, and `Income`.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a0d78a98-cfaa-4538-ad83-69c114eb6572" 
       alt="Decision Tree Visualization" 
       width="600" />
</p>

<p align="center">
  <em>Figure 6: Decision Tree Visualization</em>
</p>

### 5. Model Comparison
*   **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices were used.
*   **Performance:** 
    *   **Logistic Regression:** Achieved an ROC-AUC of 0.827, showing slightly better overall discrimination.
    *   **Decision Tree:** Achieved an ROC-AUC of 0.802. Crucially, it provided a **lift of 2.97** in response rate (44.4% vs 15.0% overall) when targeting the top 20% of customers.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5b0d4879-cbda-4b19-873e-8d53587e194a" 
       alt="Model Comparison 1" 
       width="600" />
</p>

<p align="center">
  <em>Figure 7:Model Comparison 1</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/36409131-84b2-4474-aef7-b3698f31af99" 
       alt="Model Comparison 2" 
       width="600" />
</p>

<p align="center">
  <em>Figure 8:Model Comparison 2</em>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/233206c0-0ed4-4b63-988e-80306a46ab37" 
       alt="ROC curve comparison" 
       width="250" height='auto'/>
  <img src="https://github.com/user-attachments/assets/1214064a-3d80-4406-8f44-aabf8bc7e0e9" 
       alt="ROC curve comparison" 
       width="250" height='auto'/>
</p>

<p align="center">
  <em>Figure 9:ROC curve comparison</em>
</p>

## Recommendations

1.  **Decision Tree for Targeting:** Utilize the Decision Tree for prioritizing and targeting customers due to its high lift and interpretable if-then rules, which are easy to implement in CRM systems. This will significantly improve campaign efficiency by focusing on the highest-probability responders.
2.  **Logistic Regression for Explanation:** Employ Logistic Regression for explaining and justifying marketing decisions. Its interpretable coefficients and stable probability scores offer valuable insights into *why* certain customer attributes influence response, supporting strategic planning.

The two models are complementary, with the Decision Tree driving operational targeting and Logistic Regression providing strategic understanding.

## Ethical, Privacy, and Security Considerations
*   **Ethical:** Guard against unfair targeting. Regularly review selected customer lists to ensure no groups are consistently excluded based on proxies for social advantage.
*   **Privacy:** Treat customer-level scores as sensitive data. Use minimal fields, anonymize identifiers where possible, and restrict access to campaign teams only.
*   **Security:** Implement strict access controls for customer score lists. Ensure secure storage, audit trails, and limited reuse of scores to mitigate data breach risks.
