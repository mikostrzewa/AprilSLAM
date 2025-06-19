import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

# Get the project root directory
script_dir = os.path.dirname(__file__)
project_root = os.path.join(script_dir, '..', '..')
data_csv_dir = os.path.join(project_root, 'data', 'csv')

# Replace 'your_data.csv' with your dataset file path
data_path = os.path.join(data_csv_dir, 'error_data.csv')  # Update with your file path
data = pd.read_csv(data_path)

# Define input features and performance metrics
features = [
    "Time", "Number of Nodes", "Avrg Distance",
    "Est_X", "Est_Y", "Est_Z", "Est_Roll", "Est_Pitch", "Est_Yaw"
]
performance_metrics = ["Translation Difference", "Rotation Difference"]

X = data[features]
y = data[performance_metrics]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality reduction with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Clustering with K-means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Add clusters to the data for analysis
data["Cluster"] = clusters

# Multivariable regression
regressor = LinearRegression()
regressor.fit(X, y["Translation Difference"])
y_pred = regressor.predict(X)
mse = mean_squared_error(y["Translation Difference"], y_pred)
r2 = r2_score(y["Translation Difference"], y_pred)

# Get feature importance from regression coefficients
feature_importance = np.abs(regressor.coef_)
feature_importance_normalized = feature_importance / np.sum(feature_importance)

# Visualization
# 1. Clustering visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", s=30)
plt.colorbar(label="Cluster")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA and Clustering Visualization")
plt.show()

# 2. Regression performance
plt.figure(figsize=(8, 6))
plt.scatter(y["Translation Difference"], y_pred, alpha=0.7, edgecolors="k")
plt.xlabel("Actual Translation Difference")
plt.ylabel("Predicted Translation Difference")
plt.title("Multivariable Regression: Actual vs Predicted")
plt.show()

# 3. Feature importance visualization
plt.figure(figsize=(10, 6))
plt.bar(features, feature_importance_normalized, alpha=0.7)
plt.xlabel("Features")
plt.ylabel("Normalized Importance")
plt.title("Feature Importance (Regression Coefficients)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print regression performance metrics
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save data with clusters for further analysis
output_path = os.path.join(data_csv_dir, 'slam_clustered_data.csv')
data.to_csv(output_path, index=False)

print(f"Clustered data saved to '{output_path}'.")
