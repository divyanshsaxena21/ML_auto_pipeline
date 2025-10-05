import pandas as pd
from .logging_utils import get_logger

logger = get_logger("eda")

def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.debug(f"Loaded dataframe shape={df.shape}")
    return df

def basic_report(df):
    logger.info("Generating basic EDA report")
    report = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "head": df.head().to_dict(orient='records')
    }
    logger.debug(f"Report keys={list(report.keys())}")
    return report
