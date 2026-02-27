from sklearn.metrics import silhouette_score
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_score(df_scaled):
    logger.info('evaluate silhouette score.....')
    features = df_scaled.drop(columns=['cluster', 'cluster_label'], errors='ignore')
    score = silhouette_score(features, df_scaled['cluster'])
    logger.info(f"Silhouette Score: {score:.4f}")
    return score