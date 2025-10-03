import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import silhouette_score

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
plt.savefig('.github/assets/elbow-method-plot.png')
plt.show()

# --- 3. Train and Evaluate Final Models (Custom vs. Scikit-learn) ---
print("--- Training Final Models ---")
# Your custom model
optimal_k = 5
custom_model = KMeans(K=optimal_k, max_iters=100)
custom_clusters = custom_model.fit(X_scaled)
custom_inertia = custom_model.inertia()
custom_silhouette = silhouette_score(X_scaled, custom_clusters)
custom_centroids = np.array(custom_model.centroids)

# Scikit-learn's model for comparison
sklearn_model = SklearnKMeans(n_clusters=optimal_k, n_init='auto', random_state=42)
sklearn_clusters = sklearn_model.fit_predict(X_scaled)
sklearn_inertia = sklearn_model.inertia_
sklearn_silhouette = silhouette_score(X_scaled, sklearn_clusters)
sklearn_centroids = sklearn_model.cluster_centers_

# --- 4. Print Comparison Metrics ---
print("\n--- Model Performance Comparison ---")
print(f"Metric              | Custom KMeans | Scikit-learn KMeans")
print(f"--------------------|---------------|--------------------")
print(f"Inertia (WCSS)      | {custom_inertia:<13.2f} | {sklearn_inertia:<18.2f}")
print(f"Silhouette Score    | {custom_silhouette:<13.3f} | {sklearn_silhouette:<18.3f}")


# --- 5. Visualize the Comparison ---
print("\nGenerating comparison plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
fig.suptitle('Side-by-Side Comparison of Clustering Results', fontsize=16, y=0.95)

# Plot 1: Your Custom KMeans
ax1.set_title('1. Custom KMeans Implementation', fontweight='bold')
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=custom_clusters, palette='viridis', s=100, alpha=0.8, edgecolor='k', ax=ax1)
ax1.scatter(custom_centroids[:, 0], custom_centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
ax1.set_xlabel('Annual Income (Standardized)')
ax1.set_ylabel('Spending Score (Standardized)')
ax1.legend()
ax1.grid(True)

# Plot 2: Scikit-learn's KMeans
ax2.set_title("2. Scikit-learn's KMeans Implementation", fontweight='bold')
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=sklearn_clusters, palette='viridis', s=100, alpha=0.8, edgecolor='k', ax=ax2)
ax2.scatter(sklearn_centroids[:, 0], sklearn_centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
ax2.set_xlabel('Annual Income (Standardized)')
ax2.legend()
ax2.grid(True)

plt.savefig('.github/assets/comparison-plot.png')
plt.show()
