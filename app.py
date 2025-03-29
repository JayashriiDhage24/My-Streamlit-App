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
df = pd.read_csv(url)

# Display data preview
st.write("### Sample Data")
st.dataframe(df.head())

# Handling missing values
df['Income'].fillna(df['Income'].median(), inplace=True)

# Drop unnecessary columns
df.drop(['ID', 'Year_Birth'], axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
df['Education'] = le.fit_transform(df['Education'])
df['Marital_Status'] = le.fit_transform(df['Marital_Status'])

# Normalize numerical columns
features_to_scale = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                     'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

scaler = MinMaxScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# Convert Date Column
# Convert 'Dt_Customer' while handling errors
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')

# Check if there are any NaT (missing) values after conversion
if df['Dt_Customer'].isna().sum() > 0:
    st.write("⚠️ Warning: Some dates could not be converted and will be replaced with the median date.")

# Fill missing dates with median
df['Dt_Customer'].fillna(df['Dt_Customer'].median(), inplace=True)

# Convert to ordinal format
df['Dt_Customer'] = df['Dt_Customer'].apply(lambda date: date.toordinal())


# PCA
pca = PCA()
pc_data = pca.fit_transform(df[features_to_scale])
pc_data = pd.DataFrame(pc_data)

# K-Means Clustering
k_pca = 4
kmeans = KMeans(n_clusters=k_pca, random_state=0)
df['kmeans_cluster'] = kmeans.fit_predict(pc_data.iloc[:, :2])

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
scatter = ax.scatter(pc_data.iloc[:, 0], pc_data.iloc[:, 1], c=df['kmeans_cluster'], cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clusters')
st.pyplot(fig)

# **📊 Hierarchical Clustering**
st.write("### Hierarchical Clustering")

hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
df['hierarchical_cluster'] = hierarchical.fit_predict(df[features_to_scale])

fig, ax = plt.subplots()
scatter = ax.scatter(df[features_to_scale[0]], df[features_to_scale[1]], c=df['hierarchical_cluster'], cmap='rainbow')
plt.xlabel(features_to_scale[0])
plt.ylabel(features_to_scale[1])
plt.title('Hierarchical Clustering')
st.pyplot(fig)

# Silhouette Score
silhouette_avg_hierarchical = silhouette_score(df[features_to_scale], df['hierarchical_cluster'])
st.write(f"Silhouette Score (Hierarchical Clustering): {silhouette_avg_hierarchical:.2f}")

# **📊 DBSCAN Clustering**
st.write("### DBSCAN Clustering")

dbscan = DBSCAN(eps=0.5, min_samples=5)
df['dbscan_cluster'] = dbscan.fit_predict(df[features_to_scale])

fig, ax = plt.subplots()
scatter = ax.scatter(df[features_to_scale[0]], df[features_to_scale[1]], c=df['dbscan_cluster'], cmap='rainbow')
plt.xlabel(features_to_scale[0])
plt.ylabel(features_to_scale[1])
plt.title('DBSCAN Clusters')
st.pyplot(fig)
