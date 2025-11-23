# NBA MVP Predictor

A machine learning project for predicting NBA awards winners, including MVP, DPOY, ROY, 6MOY, MIP, and more.

## Project Structure

```
mvp-predictor/
├── src/           # Source code
│   └── main.py    # Main prediction module
├── tests/         # Test files
├── data/          # Data files (training data, predictions)
├── models/        # Trained models (saved after training)
├── predict_2026_mvp.py  # Quick script to predict 2026 MVP
└── docs/          # Documentation
```

## Setup

1. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the models (this will fetch data for 2023 and 2025 seasons):
```bash
python src/main.py
```

**Note:** Always activate the virtual environment before running scripts:
```bash
source venv/bin/activate  # On macOS/Linux
```

This will:
- Fetch awards data and advanced stats from basketball-reference.com
- Create a training dataset
- Train models for MVP, DPOY, ROY, 6MOY, MIP
- Save models to the `models/` directory

## Usage

### Training Models

Train models on historical data:
```bash
python src/main.py
```

### Predicting MVP for 2026

After training, predict the 2026 MVP winner using 2026 advanced stats:

**Option 1: Using the quick script**
```bash
python predict_2026_mvp.py
```

**Option 2: Using command line arguments**
```bash
python src/main.py predict 2026
```

**Option 3: Using Python directly**
```python
from src.main import predict_mvp_for_year

predictions = predict_mvp_for_year(
    year=2026,
    models_dir="models",
    output_path="data/mvp_predictions_2026.csv"
)
```

### Predicting Other Awards

You can also predict other awards by loading the models:

```python
from src.main import NBAAwardsPredictor, fetch_advanced_stats

# Load trained models
predictor = NBAAwardsPredictor()
predictor.load_models("models")

# Fetch 2026 advanced stats
adv_stats = fetch_advanced_stats(2026)
adv_stats = predictor.preprocess_data(adv_stats)

# Predict all awards
predictions = predictor.predict_all_awards(adv_stats)
print(predictions.head(10))
```

## Features

- **Multi-Award Prediction**: Predicts MVP, DPOY, ROY, 6MOY, MIP
- **Automatic Data Fetching**: Fetches data from basketball-reference.com
- **Machine Learning Models**: Uses Random Forest classifiers
- **Future Year Prediction**: Predict awards for any year using advanced stats

## Data Sources

- Awards data: [basketball-reference.com/awards](https://www.basketball-reference.com/awards/)
- Advanced stats: [basketball-reference.com/leagues/NBA_*_advanced.html](https://www.basketball-reference.com/leagues/)

## Model Performance

After training, the script will display performance metrics for each award model including:
- Accuracy
- Precision
- Recall
- F1 Score

## GitHub Repository

This project is linked to: https://github.com/goatcodebest/NBA-MVP-Predictor.git

## Development

This project uses:
- pandas for data manipulation
- scikit-learn for machine learning
- lxml for HTML parsing

## Notes

- Models need to be trained before making predictions
- The prediction function automatically fetches advanced stats for the specified year
- Predictions are saved to CSV files in the `data/` directory

