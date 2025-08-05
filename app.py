import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import warnings

# Suppress warnings for a cleaner interface
warnings.filterwarnings('ignore')

# --- CUSTOM K-MEANS IMPLEMENTATION ---
def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2)**2))

def initialize_centroids(data, k):
    """Initializes k centroids by randomly selecting points from the data."""
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_to_clusters(data, centroids):
    """Assigns each data point to the closest centroid."""
    clusters = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_idx = np.argmin(distances)
        clusters.append(cluster_idx)
    return np.array(clusters)

def update_centroids(data, clusters, k):
    """Updates centroids by calculating the mean of the points in each cluster."""
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        points_in_cluster = data[clusters == i]
        if len(points_in_cluster) > 0:
            new_centroids[i] = np.mean(points_in_cluster, axis=0)
    return new_centroids

def custom_kmeans(data, k, max_iters=100):
    """
    Custom K-Means clustering algorithm.
    Returns the final cluster assignments and centroids.
    """
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

def run_analysis(file_obj, n_clusters):
    """
    This is the main function that runs the full pipeline:
    1. Loads and cleans the data.
    2. Engineers RFM features.
    3. Preprocesses data for clustering (log transform, scaling).
    4. Runs K-Means and generates all outputs.
    """
    if file_obj is None:
        raise gr.Error("Please upload a dataset first.")

    try:
        # --- 1. DATA CLEANING & PREPARATION ---
        df = pd.read_csv(file_obj.name, encoding='ISO-8859-1')
        
        # Remove rows with missing CustomerID
        df.dropna(axis=0, subset=['CustomerID'], inplace=True)
        
        # Remove returns (negative quantity)
        df = df[df['Quantity'] > 0]
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        
        # Convert CustomerID to integer
        df['CustomerID'] = df['CustomerID'].astype(int)
        
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Create a TotalPrice column
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

        # --- 2. RFM FEATURE ENGINEERING ---
        # Set a snapshot date for calculating recency (1 day after the last transaction)
        snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
        
        # Calculate RFM values
        rfm_df = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
            'InvoiceNo': 'nunique',
            'TotalPrice': 'sum'
        })
        
        # Rename columns for clarity
        rfm_df.rename(columns={'InvoiceDate': 'Recency', 
                               'InvoiceNo': 'Frequency', 
                               'TotalPrice': 'MonetaryValue'}, inplace=True)

        # --- 3. PREPROCESSING FOR K-MEANS ---
        rfm_log = np.log1p(rfm_df)
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_log)

        # --- 4. ELBOW METHOD ANALYSIS (using custom K-Means) ---
        inertia = []
        k_range = range(2, 11)
        for k in k_range:
            clusters, centroids = custom_kmeans(rfm_scaled, k)
            # Calculate inertia
            current_inertia = 0
            for i in range(k):
                points_in_cluster = rfm_scaled[clusters == i]
                current_inertia += np.sum((points_in_cluster - centroids[i])**2)
            inertia.append(current_inertia)

        # Create Elbow Plot
        fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
        ax_elbow.plot(k_range, inertia, 'bo-')
        ax_elbow.set_xlabel('Number of clusters (K)')
        ax_elbow.set_ylabel('Inertia')
        ax_elbow.set_title('The Elbow Method for Optimal K')
        ax_elbow.grid(True)

        # --- 5. K-MEANS CLUSTERING (using custom K-Means) ---
        n_clusters = int(n_clusters)
        clusters, _ = custom_kmeans(rfm_scaled, n_clusters)
        rfm_df['Cluster'] = clusters

        # --- 6. VISUALIZATION & SUMMARY ---
        # Create Scatter Plot
        fig_cluster, ax_cluster = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            data=rfm_df,
            x='Recency',
            y='Frequency',
            hue='Cluster',
            palette='viridis',
            s=100,
            alpha=0.7,
            ax=ax_cluster
        )
        ax_cluster.set_title(f'Customer Segments based on Recency vs Frequency')
        ax_cluster.grid(True)
        
        # Create Summary Table
        cluster_summary = rfm_df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'MonetaryValue': 'mean'
        }).round(1)
        cluster_summary['Customer Count'] = rfm_df['Cluster'].value_counts()
        cluster_summary = cluster_summary.reset_index()

        return fig_cluster, fig_elbow, cluster_summary

    except Exception as e:
        raise gr.Error(f"An error occurred during analysis: {e}")

# --- GRADIO UI DEFINITION ---
with gr.Blocks(theme=gr.themes.Soft(), title="RFM K-Means Clustering") as demo:
    gr.Markdown("# E-Commerce Customer Segmentation using RFM & K-Means")
    gr.Markdown(
        "Upload your e-commerce transaction data (`data.csv`). The app will automatically perform RFM analysis "
        "to calculate Recency, Frequency, and Monetary value for each customer, and then use K-Means to find distinct segments."
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload E-Commerce CSV File", file_types=[".csv"])
            k_slider = gr.Slider(minimum=2, maximum=10, step=1, value=4, label="How many customer segments (K)?")
            analyze_button = gr.Button("Find Customer Segments", variant="primary")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Customer Segments Plot"):
                    cluster_plot_output = gr.Plot(label="Customer Segments")
                with gr.TabItem("Optimal K (Elbow Method)"):
                    elbow_plot_output = gr.Plot(label="Elbow Method for Optimal K")
                with gr.TabItem("Segment Summary"):
                    gr.Markdown("This table shows the average RFM characteristics for each customer segment.")
                    summary_output = gr.DataFrame(label="Cluster Characteristics")
    
    analyze_button.click(
        fn=run_analysis,
        inputs=[file_input, k_slider],
        outputs=[cluster_plot_output, elbow_plot_output, summary_output]
    )

if __name__ == "__main__":
    demo.launch()