"""
Evaluate the MVP prediction model on training years (2023, 2025)
and create a confusion matrix showing which predictions were correct.
"""

from src.main import NBAAwardsPredictor, fetch_all_awards_data, fetch_advanced_stats, create_training_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_mvp_predictions(years=[2023, 2025], models_dir="models"):
    """
    Evaluate MVP predictions on training years.
    
    Args:
        years: List of years to evaluate
        models_dir: Directory containing trained models
    """
    print("="*60)
    print("Evaluating MVP Prediction Model")
    print("="*60)
    
    # Load predictor
    predictor = NBAAwardsPredictor()
    predictor.load_models(models_dir)
    
    if 'MVP' not in predictor.models:
        print("Error: MVP model not found. Please train the model first.")
        return
    
    if predictor.feature_columns is None:
        print("Error: Feature columns not found. Please train the model first.")
        return
    
    all_predictions = []
    all_actuals = []
    all_players = []
    all_years = []
    
    for year in years:
        print(f"\n{'='*60}")
        print(f"Evaluating {year} season")
        print(f"{'='*60}")
        
        # Get actual MVP winners
        awards_data = fetch_all_awards_data(year)
        actual_mvps = []
        
        if 'MVP' in awards_data and not awards_data['MVP'].empty:
            mvp_df = awards_data['MVP']
            # Get top 5 vote getters (as we did in training)
            mvp_df_clean = mvp_df.copy()
            mvp_df_clean['Player'] = mvp_df_clean['Player'].astype(str).str.replace('*', '', regex=False)
            mvp_df_clean['Player'] = mvp_df_clean['Player'].str.strip()
            actual_mvps = mvp_df_clean['Player'].head(5).tolist()
            actual_mvps = [str(w).replace('*', '').strip() for w in actual_mvps if pd.notna(w) and str(w) != 'nan']
            print(f"Actual MVP candidates (top 5): {', '.join(actual_mvps)}")
        
        # Get advanced stats for this year
        adv_stats = fetch_advanced_stats(year)
        
        if adv_stats.empty:
            print(f"  ⚠ No advanced stats for {year}")
            continue
        
        # Preprocess
        adv_stats = predictor.preprocess_data(adv_stats, filter_min_games=False)
        
        # Make predictions
        predictions_df = predictor.predict_all_awards(adv_stats)
        
        if 'MVP_probability' not in predictions_df.columns:
            print(f"  ⚠ No MVP predictions for {year}")
            continue
        
        # Get top 5 predicted MVPs
        top_predictions = predictions_df.nlargest(5, 'MVP_probability')
        predicted_mvps = top_predictions['Player'].tolist()
        print(f"Predicted MVP candidates (top 5): {', '.join(predicted_mvps)}")
        
        # Create binary labels for all players
        for idx, row in predictions_df.iterrows():
            player = row['Player']
            prob = row['MVP_probability']
            
            # Actual: 1 if in top 5 actual MVPs, 0 otherwise
            actual = 1 if any(player.replace('*', '').strip().lower() == mvp.replace('*', '').strip().lower() 
                              for mvp in actual_mvps) else 0
            
            # Predicted: 1 if in top 5 predicted, 0 otherwise
            predicted = 1 if player in predicted_mvps else 0
            
            all_predictions.append(predicted)
            all_actuals.append(actual)
            all_players.append(player)
            all_years.append(year)
            
            # Show matches
            if actual == 1:
                status = "✓ CORRECT" if predicted == 1 else "✗ MISSED"
                print(f"  {status}: {player} (actual MVP candidate, prob: {prob:.3f})")
    
    # Create confusion matrix
    print(f"\n{'='*60}")
    print("Confusion Matrix")
    print(f"{'='*60}")
    
    cm = confusion_matrix(all_actuals, all_predictions)
    
    # Create a more readable confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=['Not MVP Candidate', 'MVP Candidate (Actual)'],
        columns=['Not Predicted', 'Predicted as MVP']
    )
    
    print("\nConfusion Matrix:")
    print(cm_df)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    print(f"\nMetrics:")
    print(f"  True Positives (TP): {tp} - Correctly predicted MVP candidates")
    print(f"  True Negatives (TN): {tn} - Correctly predicted non-MVP candidates")
    print(f"  False Positives (FP): {fp} - Predicted as MVP but weren't")
    print(f"  False Negatives (FN): {fn} - Were MVP candidates but not predicted")
    
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"  Precision: {precision:.3f} ({precision*100:.1f}%)")
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"  Recall: {recall:.3f} ({recall*100:.1f}%)")
    
    if tp + tn + fp + fn > 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Show detailed results
    print(f"\n{'='*60}")
    print("Detailed Results by Player")
    print(f"{'='*60}")
    
    results_df = pd.DataFrame({
        'Year': all_years,
        'Player': all_players,
        'Actual_MVP_Candidate': all_actuals,
        'Predicted_MVP': all_predictions
    })
    
    # Show correct predictions
    correct = results_df[(results_df['Actual_MVP_Candidate'] == 1) & (results_df['Predicted_MVP'] == 1)]
    if len(correct) > 0:
        print("\n✓ Correctly Predicted MVP Candidates:")
        for _, row in correct.iterrows():
            print(f"  {row['Year']}: {row['Player']}")
    
    # Show missed predictions
    missed = results_df[(results_df['Actual_MVP_Candidate'] == 1) & (results_df['Predicted_MVP'] == 0)]
    if len(missed) > 0:
        print("\n✗ Missed MVP Candidates (were actual but not predicted):")
        for _, row in missed.iterrows():
            print(f"  {row['Year']}: {row['Player']}")
    
    # Show false positives
    false_pos = results_df[(results_df['Actual_MVP_Candidate'] == 0) & (results_df['Predicted_MVP'] == 1)]
    if len(false_pos) > 0:
        print("\n⚠ False Positives (predicted but weren't actual MVP candidates):")
        for _, row in false_pos.iterrows():
            print(f"  {row['Year']}: {row['Player']}")
    
    # Create visualization
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
        plt.title('MVP Prediction Confusion Matrix\n(Top 5 Candidates)', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix_mvp.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved to: confusion_matrix_mvp.png")
    except Exception as e:
        print(f"\n⚠ Could not create visualization: {e}")
    
    return results_df, cm_df

if __name__ == "__main__":
    # Evaluate on training years
    years = [2023, 2025]
    results, cm = evaluate_mvp_predictions(years=years)
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")

