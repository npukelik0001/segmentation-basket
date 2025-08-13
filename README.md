Online Sales Customer Analytics
Overview
This project analyzes online sales data using customer segmentation and market basket analysis. The goal is to identify customer groups with similar buying behaviors and uncover product association rules for actionable business insights.

Features
Customer Segmentation:
Uses RFM (Recency, Frequency, Monetary) analysis and clustering (KMeans and Hierarchical) to segment customers into meaningful groups with detailed personas.

Market Basket Analysis:
Applies FP-Growth algorithm to find frequent itemsets and generates association rules for cross-selling opportunities.

Interactive Web Application:
Built with Streamlit, providing dashboards for segmentation visualization, rule exploration, recommendations, and business intelligence summaries.

Data
customer_segments.csv — RFM data with cluster labels

association_rules.csv — Market basket association rules

cluster_profiles.csv — Cluster summary statistics

cleaned_customer_data.csv — Cleaned transactional data used for analysis

Usage
Clone the repository

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

Access the app locally at http://localhost:8501

Dependencies
pandas, numpy

scikit-learn

mlxtend

plotly

networkx

streamlit

Deployment
The app is deployed on Streamlit Community Cloud and accessible 24/7 at:
[YOUR_DEPLOYMENT_URL]

Business Insights
Distinct customer segments identified with tailored marketing recommendations.

Strong product association rules enable cross-selling strategies.

Estimated ROI projections provided for targeted campaigns.

Future Improvements
Incorporate real-time data updates

Add predictive modeling for customer lifetime value

Enhance recommendation engine with collaborative filtering

Author
Nastassia Pukelik — CIS9660 Data Mining Project