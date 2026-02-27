from src.utils.logger import get_logger

logger = get_logger(__name__)
def label_clusters(df_scaled):
    logger.info('clustering 4 category')
    mapping = {
        0: "High Engagement",
        1: "Moderate Engagement",
        2: "Low Engagement",
        3: "Occasional Users"
    }
    df_scaled['cluster_label'] = df_scaled['cluster'].map(mapping)
    logger.info('Cluster labeling successful ✅')
    logger.info(f"Cluster distribution:\n{df_scaled['cluster'].value_counts()}")
    return df_scaled