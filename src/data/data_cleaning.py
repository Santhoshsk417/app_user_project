import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("starting data cleaning process....")
    #filling missing values
    if "rating_given" in df.columns:
        df["rating_missing_flag"] = df["rating_given"].isnull().astype(int)# Create missing flag
        df["rating_given"] = df["rating_given"].fillna(df["rating_given"].median())# Fill with median
    return df
def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Capping outliers using IQR...")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Only cap if there are outliers
        if ((df[col] < lower) | (df[col] > upper)).any():
            df[col] = np.clip(df[col], lower, upper)        
            logger.info(f"{col}: outliers capped")

    return df