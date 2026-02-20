"""
Script to download the Iris dataset and save it to the data directory.
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


def main():
    """Download the Iris dataset and save it as CSV."""
    # Load the Iris dataset from sklearn
    iris = load_iris()
    
    # Create a DataFrame with features and target
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['target'] = iris.target
    df['species'] = df['target'].map({
        0: iris.target_names[0],
        1: iris.target_names[1],
        2: iris.target_names[2]
    })
    
    # Ensure data directory exists
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_file = data_dir / "iris.csv"
    df.to_csv(output_file, index=False)
    
    print("✓ Iris dataset downloaded successfully!")
    print(f"✓ Saved to: {output_file}")
    print(f"✓ Shape: {df.shape}")
    print(f"✓ Features: {list(iris.feature_names)}")
    print(f"✓ Classes: {list(iris.target_names)}")


if __name__ == "__main__":
    main()
