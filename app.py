import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import io
import warnings

# Suppress warnings for a cleaner interface
warnings.filterwarnings('ignore')

# --- 1. CUSTOM K-MEANS IMPLEMENTATION ---
def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2)**2))

def initialize_centroids(data, k):
    """Initializes k centroids by randomly selecting points from the data."""
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_to_clusters(data, centroids):
    """Assigns each data point to the closest centroid."""
    clusters = [np.argmin([euclidean_distance(point, centroid) for centroid in centroids]) for point in data]
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
    """Custom K-Means clustering algorithm."""
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# --- 2. GRADIO BACKEND FUNCTIONS ---

def load_initial_data(file_obj):
    """
    Loads data from an uploaded file, performs ESSENTIAL cleaning automatically,
    and prepares the initial view for the EDA tab.
    """
    if file_obj is None:
        return None, "Please upload a file first.", None, None, None
        
    df = pd.read_csv(file_obj.name, encoding='ISO-8859-1')
    initial_rows = len(df)
    
    # "Smart Defaults": Perform essential cleaning required for RFM to function.
    df.dropna(axis=0, subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df.drop_duplicates(inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    rows_after_clean = len(df)
    rows_removed = initial_rows - rows_after_clean
    status = f"Loaded '{file_obj.name}'. Performed essential cleaning (removed {rows_removed} invalid rows). Current Shape: {df.shape}"
    
    # Create the missing values plot, handling the case where there are none.
    fig, ax = plt.subplots()
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    
    if not missing_values.empty:
        missing_values.plot(kind='bar', ax=ax, figsize=(10, 6))
        ax.set_title("Remaining Optional Missing Values")
        ax.set_ylabel("Number of Missing Rows")
        plt.tight_layout()
    else:
        ax.text(0.5, 0.5, 'No Optional Missing Values Found!', ha='center', va='center', fontsize=12)
        ax.set_title("Data Cleanliness Check")
        ax.set_xticks([])
        ax.set_yticks([])

    return df, status, df.head(), df.columns.tolist(), fig

def eda_drop_na(df, columns_to_clean):
    """
    Performs optional cleaning based on user selection in the EDA tab.
    Operates on the DataFrame stored in the state.
    """
    if df is None:
        raise gr.Error("Please upload a dataset first.")
    if not columns_to_clean:
        raise gr.Error("Please select at least one column to clean.")
        
    original_rows = len(df)
    df.dropna(subset=columns_to_clean, inplace=True)
    rows_removed = original_rows - len(df)
    feedback = f"Action complete. Dropped {rows_removed} rows with missing values in {columns_to_clean}."
    return df, df.head(), feedback

def process_and_run_analysis(df, n_clusters):
    """
    The main analysis pipeline. Takes the (potentially cleaned) DataFrame from the
    state, performs RFM analysis, and runs K-Means clustering.
    """
    if df is None:
        raise gr.Error("Please upload and load a dataset first.")

    try:
        # RFM Feature Engineering
        snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
        rfm_df = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
            'InvoiceNo': 'nunique',
            'TotalPrice': 'sum'
        }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'MonetaryValue'})

        # Preprocessing & Scaling
        rfm_log = np.log1p(rfm_df[['Recency', 'Frequency', 'MonetaryValue']])
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_log)
        
        # Elbow Method Analysis
        inertia, k_range = [], range(2, 11)
        for k in k_range:
            clusters, centroids = custom_kmeans(rfm_scaled, k)
            current_inertia = sum(np.sum((rfm_scaled[clusters == i] - centroids[i])**2) for i in range(k) if len(rfm_scaled[clusters == i]) > 0)
            inertia.append(current_inertia)

        fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
        ax_elbow.plot(k_range, inertia, 'bo-')
        ax_elbow.set_xlabel('Number of clusters (K)')
        ax_elbow.set_ylabel('Inertia')
        ax_elbow.set_title('The Elbow Method for Optimal K')
        ax_elbow.grid(True)
        
        # Final Clustering
        clusters, _ = custom_kmeans(rfm_scaled, int(n_clusters))
        rfm_df['Cluster'] = clusters
        
        # Visualization: Scatter Plot
        fig_cluster, ax_cluster = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=rfm_df, x='Recency', y='Frequency', hue='Cluster', palette='viridis', s=100, alpha=0.8, ax=ax_cluster)
        ax_cluster.set_title(f'Customer Segments based on Recency vs Frequency')
        ax_cluster.grid(True)
        
        # Summary Table
        cluster_summary = rfm_df.groupby('Cluster').agg({
            'Recency': 'mean', 'Frequency': 'mean', 'MonetaryValue': 'mean'
        }).round(1)
        cluster_summary['Customer Count'] = rfm_df['Cluster'].value_counts()

        return fig_cluster, fig_elbow, cluster_summary.reset_index(), f"Analysis complete on {df.shape[0]} rows."

    except Exception as e:
        raise gr.Error(f"An error occurred during analysis: {e}")

# --- 3. GRADIO UI DEFINITION ---
with gr.Blocks(theme=gr.themes.Soft(), title="Interactive Customer Segmentation") as demo:
    gr.Markdown("# Interactive Customer Segmentation App with KMeans")
    
    # State object holds the dataframe, shared between tabs
    df_state = gr.State()

    with gr.Tabs():
        with gr.TabItem("1. Main Analysis"):
            status_textbox = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload E-Commerce CSV")
                    k_slider = gr.Slider(minimum=2, maximum=10, step=1, value=4, label="How many segments (K)?")
                    analyze_button = gr.Button("Find Customer Segments", variant="primary")
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("Customer Segments Plot"):
                            cluster_plot_output = gr.Plot()
                        with gr.TabItem("Optimal K (Elbow Method)"):
                            elbow_plot_output = gr.Plot()
                        with gr.TabItem("Segment Summary"):
                            summary_output = gr.DataFrame()

        with gr.TabItem("2. Interactive EDA Workbench"):
            gr.Markdown(
                "**Welcome to the EDA Workbench!** Essential cleaning has already been done automatically (like removing orders without a CustomerID).\n"
                "Use the tools below for optional cleaning. For example, you can remove rows with a missing 'Description' if you wish, although it is not required for this analysis."
            )
            with gr.Row(variant='panel'):
                with gr.Column(scale=1):
                    gr.Label("Handle Optional Missing Values")
                    eda_column_selector = gr.CheckboxGroup(label="Select columns to drop NA from")
                    eda_drop_na_button = gr.Button("Drop NA from Selected")
                    eda_feedback = gr.Textbox(label="Action Feedback", interactive=False)
                with gr.Column(scale=2):
                    gr.Label("Data Preview & Diagnostics")
                    # THIS IS THE LINE THAT WAS FIXED:
                    eda_df_preview = gr.DataFrame(label="Current Data Head")
                    eda_missing_plot = gr.Plot(label="Current Optional Missing Values")

    # --- 4. COMPONENT WIRING ---
    # When a file is uploaded, run load_initial_data.
    file_input.upload(
        fn=load_initial_data,
        inputs=file_input,
        outputs=[df_state, status_textbox, eda_df_preview, eda_column_selector, eda_missing_plot]
    )
    
    # The EDA 'drop na' button reads and writes to the main state.
    eda_drop_na_button.click(
        fn=eda_drop_na,
        inputs=[df_state, eda_column_selector],
        outputs=[df_state, eda_df_preview, eda_feedback]
    )

    # The main analysis button takes the CURRENT dataframe from the state.
    analyze_button.click(
        fn=process_and_run_analysis,
        inputs=[df_state, k_slider],
        outputs=[cluster_plot_output, elbow_plot_output, summary_output, status_textbox]
    )

if __name__ == "__main__":
    demo.launch()