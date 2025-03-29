import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel('marketing_campaign1.xlsx')
df.head()

import warnings
warnings.filterwarnings('ignore')

df.info()

df.isnull().sum()

df.describe()

median_income = df['Income'].median()
df['Income'].fillna(median_income, inplace=True)


df.drop(['ID', 'Year_Birth'], axis=1, inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns

df.hist(figsize=(15,12),bins=20)
plt.show()

plt.figure(figsize = (20,17))

# Calculate the number of rows and columns needed
num_cols = len(df.columns)
num_rows = (num_cols + 3) // 4  # Ensure enough rows, rounding up

for i, column in enumerate(df.columns):
    plt.subplot(num_rows, 4, i + 1)  # Adjust grid size dynamically
    sns.boxplot(df[column], vert=True)
    plt.title(f'Boxplot on the {column}')

plt.tight_layout()
plt.show()


def replace_outliers_with_median(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    # Calculate the median after clipping (or use the pre-calculated one)
    median = df[column].median()
    df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = median
    return df

# Example usage for 'Income' column (assuming the issue is there)
df = replace_outliers_with_median(df, 'Income')
df

# Correlation matrix
# Define numerical_columns
numerical_columns = df.select_dtypes(include=['number']).columns

plt.figure(figsize=(12, 8))
sns.heatmap(df[numerical_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

from sklearn.preprocessing import LabelEncoder

# Assuming 'Education' is a categorical variable you want to label encode
le = LabelEncoder()
df['Education'] = le.fit_transform(df['Education'])

df['Marital_Status'] = le.fit_transform(df['Marital_Status'])
df

from sklearn.preprocessing import MinMaxScaler

# Assuming you want to scale certain numerical features
features_to_scale = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

scaler = MinMaxScaler()

# Fit and transform the selected features
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

df


from sklearn.decomposition import PCA

# Convert 'Dt_Customer' to ordinal
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer']).apply(lambda date: date.toordinal())

pca = PCA()

pc_data = pca.fit_transform(df[features_to_scale]) # Apply PCA only to numerical features
pc_data = pd.DataFrame(pc_data)
pc_data.head()


pc_data.var()

X_new = pc_data.iloc[:,0:2]
X_new.head()

from sklearn.cluster import KMeans
inertia_pca = []
for i in range(1, 11):
    kmeans_pca = KMeans(n_clusters=i, random_state=0)
    kmeans_pca.fit(pc_data.iloc[:, :2])
    inertia_pca.append(kmeans_pca.inertia_)

plt.plot(range(1, 11), inertia_pca, marker='o')
plt.title('Elbow Method (PCA Data)')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

k_pca = 4 # Replace with your optimal k value for PCA data
kmeans_X_new = KMeans(n_clusters=k_pca, random_state=0)
# Convert column names to strings before fitting
X_new.columns = X_new.columns.astype(str)
X_new['cluster'] = kmeans_X_new.fit_predict(X_new)
X_new
# Visualize the clusters (example with first two principal components)
plt.scatter(pc_data.iloc[:, 0], pc_data.iloc[:, 1], c=X_new['cluster'], cmap='viridis')
plt.title('Clusters with PCA Data (First 4 Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

from sklearn.metrics import silhouette_score
# Print cluster statistics
print(X_new.groupby('cluster').mean())
print(X_new['cluster'].value_counts())

# Calculate Silhouette Score for X_new
silhouette_avg_X_new = silhouette_score(X_new, X_new['cluster'])
print(f"Silhouette Score (X_new): {silhouette_avg_X_new}")

from sklearn.cluster import AgglomerativeClustering
import numpy as np
# single
cluster = AgglomerativeClustering(n_clusters = 4,linkage='single')
df["single"]= cluster.fit_predict(X_new)
from sklearn.metrics import silhouette_score
c1 = silhouette_score(X_new,df["single"])
print("silhouette score-single", np.round(c1,2))

plt.figure(figsize=(6,3))
plt.scatter(X_new.iloc[:,0], X_new.iloc[:,1], c=cluster.labels_, cmap='rainbow')

# complete
cluster = AgglomerativeClustering(n_clusters = 4,linkage='complete')
df["complete"]= cluster.fit_predict(X_new)
from sklearn.metrics import silhouette_score
c2 = silhouette_score(X_new,df["complete"])
print("silhouette score-complete", np.round(c2,2))

plt.figure(figsize=(6,3))
plt.scatter(X_new.iloc[:,0], X_new.iloc[:,1], c=cluster.labels_, cmap='rainbow')


# average
cluster = AgglomerativeClustering(n_clusters = 4,linkage='average')
df["average"]= cluster.fit_predict(X_new)
from sklearn.metrics import silhouette_score
c3 = silhouette_score(X_new,df["average"])
print("silhouette score-average", np.round(c3,2))

plt.figure(figsize=(6,3))
plt.scatter(X_new.iloc[:,0], X_new.iloc[:,1], c=cluster.labels_, cmap='rainbow')


# ward
cluster = AgglomerativeClustering(n_clusters = 4,linkage='ward')
df["ward"]= cluster.fit_predict(X_new)
from sklearn.metrics import silhouette_score
c4 = silhouette_score(X_new,df["ward"])
print("silhouette score-ward", np.round(c4,2))

plt.figure(figsize=(6,3))
plt.scatter(X_new.iloc[:,0], X_new.iloc[:,1], c=cluster.labels_, cmap='rainbow')


from sklearn.cluster import AgglomerativeClustering
# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
df['hierarchical_cluster'] = hierarchical.fit_predict(df[features_to_scale])
print(df.groupby('hierarchical_cluster').mean())

plt.scatter(df[features_to_scale[0]], df[features_to_scale[1]], c=df['hierarchical_cluster'])
plt.title('Hierarchical Clusters')
plt.xlabel(features_to_scale[0])
plt.ylabel(features_to_scale[1])
plt.show()

# Calculate Silhouette Score for hierarchical clustering
silhouette_avg_hierarchical = silhouette_score(df[features_to_scale], df['hierarchical_cluster'])
print(f"Silhouette Score (Hierarchical Clustering): {silhouette_avg_hierarchical}")

# DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5) # You'll likely need to tune eps and min_samples
df['dbscan_cluster'] = dbscan.fit_predict(df[features_to_scale])
df[features_to_scale]

# Analyze DBSCAN clusters (handle -1 labels which represent noise)
print(df.groupby('dbscan_cluster').mean())

plt.scatter(df[features_to_scale[0]], df[features_to_scale[1]], c=df['dbscan_cluster'])
plt.title('DBSCAN Clusters')
plt.xlabel(features_to_scale[0])
plt.ylabel(features_to_scale[1])
plt.show()

dbscan_cluster_filtered = df[df['dbscan_cluster'] != -1]
# Check if there are at least 2 unique labels after filtering noise
unique_labels = np.unique(dbscan_cluster_filtered['dbscan_cluster'])

if len(unique_labels) >= 2:
  silhouette_avg_dbscan = silhouette_score(dbscan_cluster_filtered[features_to_scale], dbscan_cluster_filtered['dbscan_cluster'])
  print(f"Silhouette Score (DBSCAN): {silhouette_avg_dbscan}")
else:
  print("Insufficient number of clusters for Silhouette Score calculation.")

