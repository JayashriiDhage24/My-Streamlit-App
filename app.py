import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

# Load dataset from GitHub
st.title("📊 Marketing Campaign Analysis")

url = "https://raw.githubusercontent.com/JayashriiDhage24/My-Streamlit-App/main/marketing_campaign1.csv"

# Try loading dataset safely
try:
    df = pd.read_csv(url)
    st.write("✅ Data Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

# Display data preview
st.write("### Sample Data")
st.dataframe(df.head())

# Handling missing values
if "Income" in df.columns:
    df['Income'].fillna(df['Income'].median(), inplace=True)

# Drop unnecessary columns safely
drop_cols = ['ID', 'Year_Birth']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# Encode categorical columns safely
for col in ['Education', 'Marital_Status']:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convert to string before encoding

# Normalize numerical columns
features_to_scale = [
    'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'
]

scaler = MinMaxScaler()

# Only scale existing numerical columns
existing_features = [col for col in features_to_scale if col in df.columns]
df[existing_features] = scaler.fit_transform(df[existing_features])

# Convert Date Column safely
if 'Dt_Customer' in df.columns:
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
    df['Dt_Customer'].fillna(df['Dt_Customer'].median(), inplace=True)
    df['Dt_Customer'] = df['Dt_Customer'].apply(lambda date: date.toordinal())

# Fix PCA errors (handle NaN or infinite values)
df[existing_features] = df[existing_features].replace([np.inf, -np.inf], np.nan)
df.fillna(0, inplace=True)

# PCA
pca = PCA(n_components=2)  # Use 2 components for visualization
pc_data = pca.fit_transform(df[existing_features])
pc_data = pd.DataFrame(pc_data, columns=["PCA1", "PCA2"])

# K-Means Clustering
k_pca = 4
kmeans = KMeans(n_clusters=k_pca, random_state=0, n_init=10)
df['kmeans_cluster'] = kmeans.fit_predict(pc_data)

# **🎨 VISUALIZATIONS**
st.write("## 📈 Data Visualizations")

# Income Distribution
st.write("### Income Distribution")
fig, ax = plt.subplots()
sns.histplot(df["Income"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# Education Count Plot
st.write("### Education Count")
fig, ax = plt.subplots()
sns.countplot(x=df["Education"], ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# K-Means Cluster Visualization
st.write("### K-Means Clustering")
fig, ax = plt.subplots()
scatter = ax.scatter(pc_data["PCA1"], pc_data["PCA2"], c=df['kmeans_cluster'], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clusters')
st.pyplot(fig)

# **📊 Hierarchical Clustering**
st.write("### Hierarchical Clustering")

hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
df['hierarchical_cluster'] = hierarchical.fit_predict(df[existing_features])

fig, ax = plt.subplots()
scatter = ax.scatter(df[existing_features[0]], df[existing_features[1]], c=df['hierarchical_cluster'], cmap='rainbow')
plt.xlabel(existing_features[0])
plt.ylabel(existing_features[1])
plt.title('Hierarchical Clustering')
st.pyplot(fig)

# Silhouette Score
silhouette_avg_hierarchical = silhouette_score(df[existing_features], df['hierarchical_cluster'])
st.write(f"Silhouette Score (Hierarchical Clustering): {silhouette_avg_hierarchical:.2f}")

# **📊 DBSCAN Clustering**
st.write("### DBSCAN Clustering")

dbscan = DBSCAN(eps=0.5, min_samples=5)
df['dbscan_cluster'] = dbscan.fit_predict(df[existing_features])

fig, ax = plt.subplots()
scatter = ax.scatter(df[existing_features[0]], df[existing_features[1]], c=df['dbscan_cluster'], cmap='rainbow')
plt.xlabel(existing_features[0])
plt.ylabel(existing_features[1])
plt.title('DBSCAN Clusters')
st.pyplot(fig)
