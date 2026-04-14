"""
Clustering module for VRP project

Performs:
- Data loading
- K-Means clustering
- Elbow method visualization
- Map generation
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import MarkerCluster


# =========================
# Load Data
# =========================
def load_data(filepath): data/locations.xlsx
    df = pd.read_excel(filepath)

    required_columns = ['Location', 'Latitude', 'Longitude']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Excel file must contain columns: {required_columns}")

    return df


# =========================
# Elbow Method
# =========================
def plot_elbow_method(df):
    X = df[['Latitude', 'Longitude']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertia = []
    K_range = range(1, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    plt.show()

    return X_scaled


# =========================
# Run Clustering
# =========================
def run_clustering(df, X_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster Assigned'] = kmeans.fit_predict(X_scaled)

    return df


# =========================
# Save Results
# =========================
def save_results(df, output_path):
    df.to_excel(output_path, index=False)
    print(f"Clustered data saved to {output_path}")


# =========================
# Generate Map
# =========================
def generate_map(df, map_path):
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    cluster_map = folium.Map(location=map_center, zoom_start=6)

    colors = ['red', 'blue', 'green', 'purple', 'orange']

    marker_cluster = MarkerCluster().add_to(cluster_map)

    for _, row in df.iterrows():
        cluster = int(row['Cluster Assigned'])
        color = colors[cluster % len(colors)]

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            popup=f"Location: {row['Location']}, Cluster: {cluster}",
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(marker_cluster)

    cluster_map.save(map_path)
    print(f"Map saved to {map_path}")


