# Iris Classifier API

Machine Learning API for classifying Iris flowers using FastAPI and scikit-learn.

## Project Structure

```
project_iris/
├── src/
│   └── iris_api/
│       ├── __init__.py
│       └── app.py          # FastAPI application
├── scripts/
│   ├── download_data.py    # Download Iris dataset
│   └── train_iris.py       # Train classification model
├── data/
│   └── iris.csv            # Iris dataset
├── artifacts/
│   ├── iris_classifier.joblib   # Trained model
│   ├── feature_names.joblib     # Feature names
│   └── target_names.joblib      # Target class names
├── tests/
│   ├── test_health.py
│   └── test_recommend.py
├── streamlit_app.py        # Streamlit UI
├── pyproject.toml          # Project dependencies
├── Dockerfile.api          # Docker config for API
├── Dockerfile.streamlit    # Docker config for UI
├── docker-compose.yml      # Docker orchestration
└── .dockerignore           # Docker ignore patterns
```

## Setup

1. Install dependencies:
```bash
pip install -e .
```

2. Download the Iris dataset:
```bash
python scripts/download_data.py
```

3. Train the classification model:
```bash
python scripts/train_iris.py
```

## Running the Application

### 1. Start the FastAPI server:
```bash
uvicorn iris_api.app:app --reload
```

Or run directly:
```bash
python -m iris_api.app
```

The API will be available at: http://localhost:8000

### 2. Run the Streamlit UI (in a separate terminal):
```bash
streamlit run streamlit_app.py
```

The Streamlit app will open in your browser at: http://localhost:8501

## Streamlit UI Features

The web interface provides:
- 🎚️ Interactive sliders for adjusting flower measurements
- 🔮 Real-time predictions via FastAPI backend
- 📊 Visual probability distributions for all classes
- 📚 Example presets for each Iris species
- ✅ API health status monitoring

## Running with Docker

### Prerequisites
- Docker and Docker Compose installed

### Build and run with Docker Compose:

```bash
docker-compose up --build
```

This will:
- Build both the FastAPI and Streamlit containers
- Start the API on http://localhost:8000
- Start the UI on http://localhost:8501
- Automatically connect the UI to the API

### Stop the containers:
```bash
docker-compose down
```

### View logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f ui
```

### Rebuild after changes:
```bash
docker-compose up --build --force-recreate
```



## API Documentation

Once the server is running, access:
- **Interactive API docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative API docs (ReDoc)**: http://localhost:8000/redoc

## API Endpoints

### GET /
Root endpoint with API information.

### GET /health
Health check endpoint showing model status and available classes.

### POST /predict
Predict the Iris species for a single sample.

**Request body:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**
```json
{
  "predicted_class": 0,
  "predicted_species": "setosa",
  "confidence": 0.95,
  "probabilities": {
    "setosa": 0.95,
    "versicolor": 0.03,
    "virginica": 0.02
  }
}
```

### POST /predict/batch
Predict Iris species for multiple samples.

**Request body:**
```json
{
  "samples": [
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    },
    {
      "sepal_length": 6.7,
      "sepal_width": 3.0,
      "petal_length": 5.2,
      "petal_width": 2.3
    }
  ]
}
```

## Example Usage

### Using curl:
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

### Using Python requests:
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
)
print(response.json())
```

## Running Tests

```bash
pytest tests/
```

## Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: 
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Classes**: 
  - Setosa (0)
  - Versicolor (1)
  - Virginica (2)
