import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import streamlit as st
from src.utils.logger import get_logger

logger = get_logger(__name__)
def plot_clusters(df_scaled):
    logger.info('starting pca visuls...')
    # Remove non-feature columns before PCA
    features = df_scaled.drop(columns=['cluster', 'cluster_label'], errors='ignore')

    pca = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(features)
    df_scaled['PCA1'] = pca_data[:,0]
    df_scaled['PCA2'] = pca_data[:,1]
    #plot figures
    plt.figure(figsize=(8,6))
    for label in df_scaled['cluster_label'].unique():
        subset = df_scaled[df_scaled['cluster_label']==label]
        plt.scatter(subset['PCA1'], subset['PCA2'], label=label, alpha=0.6)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('User Segments')
    plt.legend()
    st.pyplot(plt)

    logger.info('PCA visualization completed ✅')