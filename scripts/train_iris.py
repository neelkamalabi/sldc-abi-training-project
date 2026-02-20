"""
Script to train an Iris classification model and save it to artifacts.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def main():
    """Train the Iris classification model."""
    # Load the data
    data_dir = Path(__file__).parent.parent / "data"
    data_file = data_dir / "iris.csv"
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print("Run 'python scripts/download_data.py' first.")
        return
    
    print(f"✓ Loading data from {data_file}")
    df = pd.read_csv(data_file)
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col not in ['target', 'species']]
    X = df[feature_columns].values
    y = df['target'].values
    
    print(f"✓ Dataset shape: {X.shape}")
    print(f"✓ Features: {feature_columns}")
    print(f"✓ Classes: {df['species'].unique().tolist()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # Train the model
    print("\n🔄 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Model trained successfully!")
    print(f"✓ Test Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=df['species'].unique().tolist()
    ))
    
    # Save the model
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = artifacts_dir / "iris_classifier.joblib"
    joblib.dump(model, model_file)
    
    print(f"✓ Model saved to: {model_file}")
    print(f"✓ Model size: {model_file.stat().st_size / 1024:.2f} KB")
    
    # Save feature names for later use
    feature_names_file = artifacts_dir / "feature_names.joblib"
    joblib.dump(feature_columns, feature_names_file)
    print(f"✓ Feature names saved to: {feature_names_file}")
    
    # Save target names
    target_names_file = artifacts_dir / "target_names.joblib"
    joblib.dump(df['species'].unique().tolist(), target_names_file)
    print(f"✓ Target names saved to: {target_names_file}")


if __name__ == "__main__":
    main()
