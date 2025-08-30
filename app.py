import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import warnings

warnings.filterwarnings('ignore')

# --- CUSTOM K-MEANS IMPLEMENTATION ---
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_to_clusters(data, centroids):
    clusters = [np.argmin([euclidean_distance(point, centroid) for centroid in centroids]) for point in data]
    return np.array(clusters)

def update_centroids(data, clusters, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        points_in_cluster = data[clusters == i]
        if len(points_in_cluster) > 0:
            new_centroids[i] = np.mean(points_in_cluster, axis=0)
    return new_centroids

def custom_kmeans(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        # safer stopping condition
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids
    return clusters, centroids

# --- DATA LOADING & CLEANING ---
def load_initial_data(file_obj):
    if file_obj is None:
        return None, "Please upload a file first.", None

    # safer file read
    df = pd.read_csv(file_obj, encoding='ISO-8859-1')
    initial_rows = len(df)
    df.dropna(axis=0, subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df.drop_duplicates(inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    rows_after_clean = len(df)
    rows_removed = initial_rows - rows_after_clean
    status = f"Loaded file. Removed {rows_removed} invalid rows. Current Shape: {df.shape}"
    return df, status, df

# --- ELBOW PLOT + EXTRA EDA ---
def compute_elbow_and_eda(df):
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm_df = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate':'Recency','InvoiceNo':'Frequency','TotalPrice':'MonetaryValue'})

    # --- Histograms ---
    fig_hist, axes = plt.subplots(1, 3, figsize=(15, 4))
    rfm_df['Recency'].plot(kind='hist', bins=30, ax=axes[0], color='skyblue', edgecolor='black')
    axes[0].set_title("Recency Distribution")
    rfm_df['Frequency'].plot(kind='hist', bins=30, ax=axes[1], color='lightgreen', edgecolor='black')
    axes[1].set_title("Frequency Distribution")
    rfm_df['MonetaryValue'].plot(kind='hist', bins=30, ax=axes[2], color='salmon', edgecolor='black')
    axes[2].set_title("Monetary Value Distribution")
    plt.tight_layout()
    plt.close(fig_hist)

    # --- Correlation Heatmap ---
    fig_corr, ax_corr = plt.subplots(figsize=(6,5))
    sns.heatmap(rfm_df[['Recency','Frequency','MonetaryValue']].corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Correlation Heatmap")
    plt.close(fig_corr)

    # --- Log transform + Scaling ---
    rfm_log = np.log1p(rfm_df[['Recency','Frequency','MonetaryValue']])
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    # --- Elbow Plot ---
    inertia, k_range = [], range(2, 11)
    for k in k_range:
        clusters, centroids = custom_kmeans(rfm_scaled, k)
        current_inertia = sum(np.sum((rfm_scaled[clusters == i] - centroids[i])**2) for i in range(k))
        inertia.append(current_inertia)

    fig_elbow, ax_elbow = plt.subplots(figsize=(8,5))
    ax_elbow.plot(k_range, inertia, 'bo-')
    ax_elbow.set_xlabel('Number of clusters (K)')
    ax_elbow.set_ylabel('Inertia')
    ax_elbow.set_title('The Elbow Method for Optimal K')
    ax_elbow.grid(True)
    plt.close(fig_elbow)
    
    return fig_elbow, fig_hist, fig_corr, rfm_scaled, rfm_df

# --- MAIN ANALYSIS: CUSTOMER SEGMENTS ---
def generate_segments(rfm_scaled, rfm_df, k):
    clusters, _ = custom_kmeans(rfm_scaled, int(k))
    rfm_df['Cluster'] = clusters

    # Scatter plot
    fig_cluster, ax_cluster = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=rfm_df, x='Recency', y='Frequency', hue='Cluster',
                    palette='viridis', s=100, alpha=0.8, ax=ax_cluster)
    ax_cluster.set_title(f'Customer Segments based on Recency vs Frequency')
    ax_cluster.grid(True)
    plt.close(fig_cluster)

    # Cluster summary
    cluster_summary = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean', 'Frequency': 'mean', 'MonetaryValue': 'mean'
    }).round(1)
    cluster_summary['Customer Count'] = rfm_df.groupby('Cluster').size().values
    cluster_summary = cluster_summary.reset_index()

    return fig_cluster, cluster_summary

# --- GRADIO UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="Interactive Customer Segmentation") as demo:
    gr.Markdown("# Interactive Customer Segmentation App with KMeans + EDA")

    df_state = gr.State()
    rfm_scaled_state = gr.State()
    rfm_df_state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload E-Commerce CSV")
            load_button = gr.Button("Load Data & Run EDA", variant="primary")
            k_slider = gr.Slider(minimum=2, maximum=10, step=1, value=4, label="Adjust Number of Segments (K)")
            analyze_button = gr.Button("Generate Customer Segments", variant="primary")

        with gr.Column(scale=2):
            elbow_plot_output = gr.Plot(label="Optimal K (Elbow Method)")
            hist_output = gr.Plot(label="RFM Histograms")
            corr_output = gr.Plot(label="Correlation Heatmap")
            cluster_plot_output = gr.Plot(label="Customer Segments Plot")
            summary_output = gr.DataFrame(label="Segment Summary")
            status_textbox = gr.Textbox(label="Status", interactive=False)

    # --- FUNCTION WIRING ---
    def load_and_show_eda(file_obj):
        df, status, df_cleaned = load_initial_data(file_obj)
        if df is None:
            return None, None, None, None, None, status
        fig_elbow, fig_hist, fig_corr, rfm_scaled, rfm_df = compute_elbow_and_eda(df_cleaned)
        return df, fig_elbow, fig_hist, fig_corr, rfm_scaled, rfm_df, status

    load_button.click(
        fn=load_and_show_eda,
        inputs=file_input,
        outputs=[df_state, elbow_plot_output, hist_output, corr_output, rfm_scaled_state, rfm_df_state, status_textbox]
    )

    analyze_button.click(
        fn=generate_segments,
        inputs=[rfm_scaled_state, rfm_df_state, k_slider],
        outputs=[cluster_plot_output, summary_output]
    )

if __name__ == "__main__":
    demo.launch()

