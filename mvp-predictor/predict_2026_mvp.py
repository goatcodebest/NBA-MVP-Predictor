"""
Quick script to predict 2026 MVP winner using trained models.
Run this after training the models with: python src/main.py
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import predict_mvp_for_year

if __name__ == "__main__":
    # Predict MVP for 2026
    predictions = predict_mvp_for_year(
        year=2026,
        models_dir="models",
        output_path="data/mvp_predictions_2026.csv"
    )
    
    print("\n" + "="*60)
    print("Prediction Complete!")
    print("="*60)
    print(f"\nPredicted MVP Winner: {predictions.iloc[0]['Player']}")
    print(f"Probability: {predictions.iloc[0]['MVP_probability']:.4f}")

