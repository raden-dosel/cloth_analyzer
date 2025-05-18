import pandas as pd
from prefect import flow, task
from data.preprocessing import clean_text, validate_schema
from data.scripts.generate_synthetic_data import generate_dataset

@task(name="Extract Raw Data")
def extract(raw_path: str = "data/raw"):
    """Load raw chat transcripts"""
    return pd.read_csv(f"{raw_path}/user_inputs.csv")

@task(name="Transform Data")
def transform(df: pd.DataFrame):
    """Clean and enrich raw data"""
    df["cleaned_text"] = df["user_input"].apply(clean_text)
    df = validate_schema(df)
    return df

@task(name="Generate Synthetic Data")
def augment(df: pd.DataFrame, samples: int = 1000):
    """Expand dataset with synthetic examples"""
    synthetic = generate_dataset(samples)
    return pd.concat([df, synthetic])

@flow(name="Data Preparation Pipeline")
def data_pipeline():
    # Extract
    raw_data = extract()
    
    # Transform
    cleaned_data = transform(raw_data)
    
    # Augment
    full_dataset = augment(cleaned_data)
    
    # Load
    full_dataset.to_parquet("data/processed/full_dataset.parquet")

if __name__ == "__main__":
    data_pipeline()