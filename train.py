import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Import your custom model
from kmeans import KMeans

# --- 1. Load and Prepare the Data ---
df = pd.read_csv('data/Mall_Customers.csv')
# Select relevant features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. (Additional Feature) Find the Optimal K using the Elbow Method ---
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans_scratch = KMeans(K=k, max_iters=100)
    kmeans_scratch.fit(X_scaled)
    wcss.append(kmeans_scratch.inertia())

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, 'bo-')
plt.title('Elbow Method to Find Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# --- 3. Train Final Model with Optimal K ---
# From the plot, the elbow is at K=5.
optimal_k = 5
final_model = KMeans(K=optimal_k, max_iters=100)
clusters = final_model.fit(X_scaled)

# --- 4. Visualize the Final Clusters ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette='viridis', s=100, alpha=0.7, edgecolor='k')
# Plot the centroids
centroids = np.array(final_model.centroids)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')

plt.title(f'Customer Segments (K={optimal_k})')
plt.xlabel('Annual Income (Standardized)')
plt.ylabel('Spending Score (Standardized)')
plt.legend()
plt.grid(True)
plt.show()