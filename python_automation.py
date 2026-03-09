import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

print("Pipeline Started...")

# -----------------------------
# 1. Load Cleaned Data
# -----------------------------
customers = pd.read_csv("../Data/Cleaned data/customers.csv")
orders = pd.read_csv("../Data/Cleaned data/orders.csv")
products = pd.read_csv("../Data/Cleaned data/products.csv")

print("Data Loaded Successfully")

# -----------------------------
# 2. Merge Datasets
# -----------------------------
df = orders.merge(products, on="product_id")

print("Datasets Merged")

# -----------------------------
# 3. Load RFM Base
# -----------------------------
rfm = pd.read_csv("../Data/Cleaned data/rfm_base.csv")

rfm['last_purchase'] = pd.to_datetime(rfm['last_purchase'])

reference_date = rfm['last_purchase'].max()

rfm['recency'] = (reference_date - rfm['last_purchase']).dt.days

print("Recency Calculated")

# -----------------------------
# 4. RFM Scoring
# -----------------------------
rfm['R_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1])
rfm['F_score'] = pd.qcut(rfm['frequency'], 5, labels=[1,2,3,4,5])
rfm['M_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5])

rfm['R_score'] = rfm['R_score'].astype(int)
rfm['F_score'] = rfm['F_score'].astype(int)
rfm['M_score'] = rfm['M_score'].astype(int)

print("RFM Scores Generated")

# -----------------------------
# 5. Customer Segmentation
# -----------------------------
def segment(row):
    if row['R_score'] >= 4 and row['F_score'] >= 4 and row['M_score'] >= 4:
        return 'Champions'
    elif row['R_score'] >= 3 and row['F_score'] >= 3:
        return 'Loyal Customers'
    elif row['R_score'] <= 2 and row['F_score'] >= 3:
        return 'At Risk'
    elif row['R_score'] == 1:
        return 'Lost Customers'
    else:
        return 'Potential'

rfm['Customer_Type'] = rfm.apply(segment, axis=1)

print("Customer Segmentation Completed")

# -----------------------------
# 6. Save Final RFM Dataset
# -----------------------------
rfm.to_csv("../Data/Cleaned data/rfm_final.csv", index=False)

print("RFM Data Saved")

# -----------------------------
# 7. Market Basket Analysis
# -----------------------------
basket = pd.crosstab(df['order_id'], df['product_name'])

basket = basket.apply(lambda x: x.apply(lambda y: 1 if y > 0 else 0))

frequent_items = apriori(basket, min_support=0.02, use_colnames=True)

rules = association_rules(frequent_items, metric="lift", min_threshold=1)

rules.to_csv("../Data/Cleaned data/association_rules.csv", index=False)

print("Market Basket Analysis Completed")

print("Pipeline Finished Successfully")