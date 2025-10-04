import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def basic_report(df):
    report = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "head": df.head().to_dict(orient='records')
    }
    return report
