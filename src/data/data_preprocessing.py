import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)
def preprocess_data(
    df: pd.DataFrame,
    #drop_columns: list = ['user_id', 'rating_missing_flag'],
    domain_drop_cols: list = ['churn_risk_score','engagement_score'],
    categorical_cols: list = ['gender', 'country', 'device_type', 'subscription_type', 'marketing_source'],
    heatmap_dir: str = "artifacts/reports"
) -> pd.DataFrame:
    logger.info("starting data preprocessing......")
    # 1️⃣ Drop ID and unwanted columns
    df_selected = df.drop(columns=['user_id','rating_missing_flag'])
    # 2️⃣  Select Only Numeric Columns (for clustering)
    numeric_cols = df_selected.select_dtypes(include=['int64','float64']).columns
    df_selected = df_selected[numeric_cols]
    # 3️⃣ Save initial correlation heatmap
    Path(heatmap_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(20,18))
    sns.heatmap(df_selected.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap - Before VarianceThreshold & Domain Drop")
    plt.savefig(f"{heatmap_dir}/heatmap_before.png")
    plt.close()
    # 4️⃣ Remove zero-variance columns
    selector = VarianceThreshold(threshold=0)
    selector.fit(df_selected)
    df_selected = df_selected.loc[:, selector.get_support()]
    # 5️⃣ Drop domain-specific columns
    df_selected = df_selected.drop(columns=['churn_risk_score','engagement_score'])
    # 6️⃣ Save heatmap after drops
    plt.figure(figsize=(15,12))
    sns.heatmap(df_selected.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap - After VarianceThreshold & Domain Drop")
    plt.savefig(f"{heatmap_dir}/heatmap_after.png")
    plt.close()

    logger.info(f"Saving heatmap before drops to {heatmap_dir}/heatmap_before.png")
    
    # 7️⃣ One-hot encode categorical variables
    df_cat = df[['gender', 'country', 'device_type', 'subscription_type', 'marketing_source']]
    df_cat_encoded = pd.get_dummies(df_cat, drop_first=True)
    # 8️⃣Keep only numeric columns for clustering
    numeric_cols = df_selected.select_dtypes(include=['int64', 'float64']).columns
    df_numeric = df_selected[numeric_cols]
    # 9️⃣ Combine numeric + categorical features
    df_final = pd.concat([df_numeric, df_cat_encoded], axis=1)
    
    return df_final
