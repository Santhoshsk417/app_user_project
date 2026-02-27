import yaml
import pandas as pd
from src.utils.logger import get_logger
from sklearn.decomposition import PCA

# Data & preprocessing
from src.data.data_load import load_csv
from src.data.data_cleaning import clean_data, cap_outliers
from src.data.data_preprocessing import preprocess_data
from src.data.feature_scaling import scale_features

# Clustering
from src.models.train import train_kmeans
from src.models.cluster_labelling import label_clusters
from src.models.evaluate import evaluate_score
from src.models.model_registry import save_model

logger = get_logger(__name__)
def main():
    logger.info("Starting Clustering Pipeline...")

    # 0️⃣ Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1️⃣ Load raw data
    df = load_csv(config["data"]["raw_path"])

    # 2️⃣ Clean & cap outliers
    df = clean_data(df)
    df = cap_outliers(df)

    # 3️⃣ Preprocess: drop columns, one-hot encoding
    df_final = preprocess_data(df)
    logger.info(f"Data after preprocessing: {df_final.shape}")

    # 4️⃣ Scale numeric features
    df_scaled = scale_features(df_final)
    logger.info(f"Data after scaling: {df_scaled.shape}")

    # 5️⃣ Train KMeans clusters
    df_clustered = train_kmeans(df_scaled)
    logger.info("KMeans clustering completed ✅")

    # 6️⃣ Label clusters with human-readable names
    df_clustered = label_clusters(df_clustered)
    logger.info("Cluster labeling completed ✅")

    # 7️⃣ Evaluate silhouette score
    score = evaluate_score(df_clustered)
    logger.info(f"Silhouette Score: {score:.4f}")


     # 8️⃣ Compute PCA for visualization
    pca = PCA(n_components=config["preprocessing"]["pca_components"], random_state=42)
    features_for_pca = df_clustered.drop(columns=['cluster', 'cluster_label'], errors='ignore')
    pca_data = pca.fit_transform(features_for_pca)
    df_clustered['PCA1'] = pca_data[:, 0]
    df_clustered['PCA2'] = pca_data[:, 1]
    logger.info("PCA columns added to clustered data ✅")

    # 9️⃣ Save KMeans model
    # ⚠️ Optionally, modify train_kmeans to return the actual KMeans object if you want to save it
    save_model(
        model=None,
        path=config["artifacts"]["model_path"],
        logger=logger
    )

    # 🔟 Save clustered dataframe (with cluster_label + PCA1/PCA2)
    df_clustered.to_csv(config["data"]["processed_path"], index=False)
    logger.info(f"Clustered data saved to {config['data']['processed_path']} ✅")

    logger.info("🎉 Clustering pipeline finished successfully!")

if __name__ == "__main__":
    main()

    '''# 8️⃣ Save KMeans model
    save_model(
        model=None,  # If you want to save actual kmeans object, modify train_kmeans to return model
        path=config["artifacts"]["model_path"],
        logger=logger
    )

    # 9️⃣ Save clustered dataframe
    df_clustered.to_csv(config["data"]["processed_path"], index=False)
    logger.info(f"Clustered data saved to {config['data']['processed_path']} ✅")

    logger.info("Clustering pipeline finished successfully!")

if __name__ == "__main__":
    main()'''