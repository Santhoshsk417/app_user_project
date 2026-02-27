from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)
def scale_features(df_final: pd.DataFrame, numeric_cols=None):
    logger.info("Scaling features...")
    # 1️⃣ Select numeric columns
    X = df_final.select_dtypes(include=['float64','int64'])
    # 2️⃣ Drop cluster/PCA columns if they exist
    X = X.drop(columns=['cluster', 'cluster_label', 'PCA1', 'PCA2'], errors='ignore')
    # 3️⃣ Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 4️⃣ Convert back to DataFrame
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    logger.info("Feature scaling completed")

    return df_scaled























    '''if numeric_cols is None:
        numeric_cols = df_final.select_dtypes(include=['int64','float64']).columns
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_cols])
    return pd.DataFrame(scaled, columns=numeric_cols, index=df.index), scaler'''