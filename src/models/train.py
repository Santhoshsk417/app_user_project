import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from src.utils.logger import get_logger

logger = get_logger(__name__)
def train_kmeans(df_scaled):
    logger.info('initial kmeans')
    kmeans = KMeans(n_clusters=4,random_state=42)
    clusters = kmeans.fit_predict(df_scaled)
    
    df_scaled['cluster'] = clusters
    logger.info('initiated succesfull')
    return df_scaled