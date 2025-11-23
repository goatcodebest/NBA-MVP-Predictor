"""
Main entry point for NBA Awards Predictor.
Predicts all NBA awards: MVP, DPOY, ROY, 6MOY, MIP, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')


class NBAAwardsPredictor:
    """Main class for predicting all NBA awards."""
    
    # Award names and their table indices (may vary, but typically in this order)
    AWARD_NAMES = [
        'MVP', 'DPOY', 'ROY', '6MOY', 'MIP', 'COY'
    ]
    
    def __init__(self):
        """Initialize the NBA Awards Predictor."""
        self.models = {}  # Dictionary to store models for each award
        self.scalers = {}  # Dictionary to store scalers for each award
        self.feature_columns = None
        self.data = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the data file (CSV, JSON, etc.)
            
        Returns:
            DataFrame containing the loaded data
        """
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                self.data = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, df: pd.DataFrame, filter_min_games: bool = True, 
                       min_games: int = 41, min_mp: float = 20.0) -> pd.DataFrame:
        """
        Preprocess the data for training/prediction.
        
        Args:
            df: Input DataFrame
            filter_min_games: Whether to filter out players with insufficient games/minutes
            min_games: Minimum games played (default 41, half season)
            min_mp: Minimum minutes per game (default 20.0)
            
        Returns:
            Preprocessed DataFrame
        """
        processed_df = df.copy()
        
        # Remove duplicate header rows and non-player rows
        processed_df = processed_df[processed_df['Player'] != 'Player']
        # Remove "League Average" and other non-player entries
        processed_df = processed_df[~processed_df['Player'].str.contains('League Average', case=False, na=False)]
        processed_df = processed_df[~processed_df['Player'].str.contains('Team', case=False, na=False)]
        
        # Convert numeric columns, handling errors
        numeric_cols = processed_df.select_dtypes(include=[object]).columns
        for col in numeric_cols:
            if col not in ['Player', 'Tm', 'Pos']:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # Filter players with insufficient playing time (for MVP predictions)
        if filter_min_games:
            # Check for games played column (could be 'G' or 'GP')
            games_col = None
            for col in ['G', 'GP', 'Games']:
                if col in processed_df.columns:
                    games_col = col
                    break
            
            # Check for minutes per game column (advanced stats uses 'MP' for total minutes, need to calculate MPG)
            mp_col = None
            mp_total_col = None
            for col in ['MP', 'MP_per_G', 'MP/G']:
                if col in processed_df.columns:
                    mp_col = col
                    break
            # Also check for total minutes to calculate MPG
            if 'MP' in processed_df.columns and 'G' in processed_df.columns:
                mp_total_col = 'MP'
            
            initial_count = len(processed_df)
            
            # Filter by games played (only if column exists and has valid data)
            if games_col and games_col in processed_df.columns:
                # Only filter if we have enough players with sufficient games
                # For predictions, use adaptive threshold based on data
                games_data = processed_df[games_col].dropna()
                if len(games_data) > 0:
                    max_games = games_data.max()
                    # If season is early, lower the threshold
                    if max_games < min_games:
                        adaptive_min_games = max(1, int(max_games * 0.3))  # 30% of max games
                        print(f"  Season appears early (max games: {max_games}), using adaptive threshold: {adaptive_min_games} games")
                        processed_df = processed_df[
                            (processed_df[games_col] >= adaptive_min_games) | 
                            (processed_df[games_col].isna())
                        ]
                    else:
                        processed_df = processed_df[
                            (processed_df[games_col] >= min_games) | 
                            (processed_df[games_col].isna())
                        ]
            
            # Filter by minutes per game (skip for predictions, only used during training filtering)
            # Note: We don't filter by MPG for predictions, only by total MP
            
            filtered_count = len(processed_df)
            if initial_count != filtered_count:
                print(f"  Filtered {initial_count - filtered_count} players, {filtered_count} remaining")
        
        # Fill NaN values with 0 for numeric columns (but only after filtering)
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(0)
        
        return processed_df
    
    def train_model(self, award_name: str, features: List[str], target: str, 
                   model_type: str = 'random_forest') -> Dict[str, float]:
        """
        Train a model for a specific award.
        
        Args:
            award_name: Name of the award (e.g., 'MVP', 'DPOY')
            features: List of feature column names
            target: Target column name (binary: 1 if won, 0 if not)
            model_type: Type of model ('random_forest' or 'gradient_boosting')
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Prepare features and target
        X = self.data[features].copy()
        y = self.data[target].copy()
        
        # Remove rows with NaN in target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            print(f"No valid data for {award_name}")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Store model and scaler
        self.models[award_name] = model
        self.scalers[award_name] = scaler
        
        print(f"\n{award_name} Model Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        
        return metrics
    
    def predict(self, input_data: pd.DataFrame, award_name: str) -> np.ndarray:
        """
        Make predictions for a specific award.
        
        Args:
            input_data: DataFrame containing feature values
            award_name: Name of the award to predict
            
        Returns:
            Array of predictions (probabilities)
        """
        if award_name not in self.models:
            raise ValueError(f"No model trained for {award_name}")
        
        model = self.models[award_name]
        scaler = self.scalers[award_name]
        
        # Use the same features as training
        if self.feature_columns:
            # Create missing columns (like MPG_calc) if needed
            X = input_data.copy()
            
            # Add MPG_calc if model expects it but it's not in data
            if 'MPG_calc' in self.feature_columns and 'MPG_calc' not in X.columns:
                if 'MP' in X.columns and 'G' in X.columns:
                    X['MPG_calc'] = X['MP'] / X['G'].replace(0, 1)
                else:
                    X['MPG_calc'] = 0.0
            
            # Only select features that exist in both
            available_features = [f for f in self.feature_columns if f in X.columns]
            missing_features = [f for f in self.feature_columns if f not in X.columns]
            
            if missing_features:
                print(f"  Warning: Missing features {missing_features}, filling with 0")
                for feat in missing_features:
                    X[feat] = 0.0
            
            X = X[self.feature_columns].copy()
        else:
            X = input_data.copy()
        
        X_scaled = scaler.transform(X)
        predictions = model.predict_proba(X_scaled)[:, 1]  # Probability of winning
        
        return predictions
    
    def predict_all_awards(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for all awards.
        
        Args:
            input_data: DataFrame containing feature values and Player names
            
        Returns:
            DataFrame with predictions for each award
        """
        if 'Player' in input_data.columns:
            results = input_data[['Player']].copy()
        else:
            results = pd.DataFrame()
        
        for award_name in self.models.keys():
            try:
                predictions = self.predict(input_data, award_name)
                results[f'{award_name}_probability'] = predictions
            except Exception as e:
                print(f"Warning: Could not predict {award_name}: {e}")
                results[f'{award_name}_probability'] = 0.0
        
        # Sort by MVP probability by default
        if 'MVP_probability' in results.columns:
            results = results.sort_values('MVP_probability', ascending=False)
        
        return results
    
    def save_models(self, directory: str = "models"):
        """Save all trained models and scalers."""
        os.makedirs(directory, exist_ok=True)
        
        for award_name, model in self.models.items():
            model_path = os.path.join(directory, f"{award_name}_model.pkl")
            scaler_path = os.path.join(directory, f"{award_name}_scaler.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[award_name], f)
        
        # Save feature columns
        self.save_feature_columns(directory)
        
        print(f"\nModels saved to {directory}/")
    
    def load_models(self, directory: str = "models"):
        """Load all trained models and scalers."""
        for award_name in self.AWARD_NAMES:
            model_path = os.path.join(directory, f"{award_name}_model.pkl")
            scaler_path = os.path.join(directory, f"{award_name}_scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with open(model_path, 'rb') as f:
                    self.models[award_name] = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scalers[award_name] = pickle.load(f)
        
        # Load feature columns if available
        feature_path = os.path.join(directory, "feature_columns.pkl")
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
    
    def save_feature_columns(self, directory: str = "models"):
        """Save feature columns used for training."""
        os.makedirs(directory, exist_ok=True)
        if self.feature_columns:
            feature_path = os.path.join(directory, "feature_columns.pkl")
            with open(feature_path, 'wb') as f:
                pickle.dump(self.feature_columns, f)


def fetch_all_awards_data(year: int) -> Dict[str, pd.DataFrame]:
    """
    Fetch all awards data from basketball-reference.com for a given year.
    
    Args:
        year: NBA season year
        
    Returns:
        Dictionary mapping award names to DataFrames
    """
    print(f"\nFetching all awards data for {year}...")
    
    awards_url = f"https://www.basketball-reference.com/awards/awards_{year}.html"
    
    try:
        tables = pd.read_html(awards_url, header=1)
        awards_data = {}
        
        # Award detection based on table characteristics
        # MVP is ALWAYS the first table with voting columns
        for idx, table in enumerate(tables):
            if 'Player' not in table.columns:
                continue
            
            # Clean the table
            table = table[table['Player'] != 'Player']  # Remove duplicate headers
            
            if len(table) == 0:
                continue
            
            # Try to identify award by column names or table position
            award_name = None
            cols_lower = [str(c).lower() for c in table.columns]
            cols_str = ' '.join(cols_lower)
            
            # MVP is ALWAYS table 0 and has voting columns (Pts Won, Share, First, etc.)
            if idx == 0 and ('pts won' in cols_str or 'share' in cols_str or 'first' in cols_str):
                award_name = 'MVP'
            # ROY is table 1 and has voting columns (rookies are usually young, 19-22 years old)
            elif idx == 1 and ('pts won' in cols_str or 'share' in cols_str or 'first' in cols_str):
                # Check if players are likely rookies (young age)
                if 'Age' in table.columns:
                    ages = pd.to_numeric(table['Age'], errors='coerce').dropna()
                    if len(ages) > 0 and ages.mean() < 22:  # Rookies are typically young
                        award_name = 'ROY'
                    else:
                        award_name = 'DPOY'  # Otherwise it's DPOY
                else:
                    award_name = 'ROY'  # Default to ROY if no age column
            # DPOY is typically table 2 or 3
            elif idx == 2 and ('pts won' in cols_str or 'share' in cols_str):
                award_name = 'DPOY'
            # 6MOY is typically table 3 or 4
            elif idx == 3 and ('pts won' in cols_str or 'share' in cols_str):
                award_name = '6MOY'
            # MIP is typically table 4
            elif idx == 4 and ('pts won' in cols_str or 'share' in cols_str):
                award_name = 'MIP'
            
            if award_name:
                awards_data[award_name] = table
                # Show first player for verification
                first_player = table.iloc[0]['Player'] if len(table) > 0 and 'Player' in table.columns else 'N/A'
                print(f"  ✓ {award_name}: {len(table)} players (first: {first_player})")
        
        return awards_data
    except Exception as e:
        print(f"Error fetching awards data: {e}")
        return {}


def fetch_advanced_stats(year: int) -> pd.DataFrame:
    """
    Fetch advanced stats from basketball-reference.com for a given year.
    
    Args:
        year: NBA season year
        
    Returns:
        DataFrame containing advanced stats
    """
    print(f"Fetching advanced stats for {year}...")
    
    adv_url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    
    try:
        adv = pd.read_html(adv_url, header=0)[0]
        adv = adv[adv['Player'] != 'Player']  # Remove duplicate header rows
        
        # Clean player names (remove asterisks and extra characters)
        if 'Player' in adv.columns:
            adv['Player'] = adv['Player'].str.replace('*', '', regex=False)
            adv['Player'] = adv['Player'].str.strip()
        
        print(f"  ✓ Advanced stats: {len(adv)} players")
        return adv
    except Exception as e:
        print(f"Error fetching advanced stats: {e}")
        return pd.DataFrame()


def create_training_dataset(years: List[int], output_path: str = "data/training_data.csv") -> pd.DataFrame:
    """
    Create a comprehensive training dataset by fetching awards and stats for multiple years.
    
    Args:
        years: List of years to fetch data for
        output_path: Path where the dataset will be saved
        
    Returns:
        DataFrame containing the complete training dataset
    """
    all_data = []
    
    for year in years:
        print(f"\n{'='*60}")
        print(f"Processing year {year}")
        print(f"{'='*60}")
        
        # Fetch awards data
        awards_data = fetch_all_awards_data(year)
        
        # Fetch advanced stats
        adv_stats = fetch_advanced_stats(year)
        
        if adv_stats.empty:
            print(f"  ⚠ Skipping {year} - no advanced stats available")
            continue
        
        # Create a base dataset from advanced stats
        year_data = adv_stats.copy()
        year_data['Year'] = year
        
        # Clean player names in year_data
        year_data['Player'] = year_data['Player'].str.replace('*', '', regex=False)
        year_data['Player'] = year_data['Player'].str.strip()
        
        # Add award labels (1 if won, 0 if not)
        for award_name, award_df in awards_data.items():
            if not award_df.empty and 'Player' in award_df.columns:
                # Get all players who received votes (for MVP, get top vote getters)
                # For most awards, winner is typically first row
                # Clean player names
                award_df_clean = award_df.copy()
                award_df_clean['Player'] = award_df_clean['Player'].astype(str).str.replace('*', '', regex=False)
                award_df_clean['Player'] = award_df_clean['Player'].str.strip()
                
                # For MVP, we might want to include top vote getters
                # For other awards, typically just the winner
                if award_name == 'MVP':
                    # Include top 5 vote getters as potential winners
                    winners = award_df_clean['Player'].head(5).tolist()
                else:
                    # Just the winner (first row)
                    winners = award_df_clean['Player'].head(1).tolist()
                
                winners_clean = [str(w).replace('*', '').strip() for w in winners if pd.notna(w) and str(w) != 'nan']
                
                # Create binary label with fuzzy matching
                def check_winner(player_name):
                    player_clean = str(player_name).replace('*', '').strip()
                    # Exact match
                    if player_clean in winners_clean:
                        return 1
                    # Check if any winner name is contained in player name or vice versa
                    for winner in winners_clean:
                        if player_clean.lower() == winner.lower():
                            return 1
                        # Handle cases like "LeBron James" vs "LeBron James (LAL)"
                        if player_clean.lower().startswith(winner.lower()) or winner.lower().startswith(player_clean.lower()):
                            return 1
                    return 0
                
                year_data[f'won_{award_name}'] = year_data['Player'].apply(check_winner)
            else:
                year_data[f'won_{award_name}'] = 0
        
        all_data.append(year_data)
    
    # Combine all years
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        combined_data.to_csv(output_path, index=False)
        print(f"\n{'='*60}")
        print(f"Training dataset saved to {output_path}")
        print(f"Dataset shape: {combined_data.shape}")
        print(f"{'='*60}")
        
        return combined_data
    else:
        print("No data collected!")
        return pd.DataFrame()


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns (exclude player info and target columns).
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of feature column names
    """
    exclude_cols = ['Player', 'Tm', 'Pos', 'Year', 'Rk'] + \
                   [col for col in df.columns if col.startswith('won_')]
    
    # Get numeric columns
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and 
                   df[col].dtype in [np.int64, np.float64]]
    
    return feature_cols


def main():
    """Main function."""
    print("NBA Awards Predictor")
    print("=" * 60)
    
    # Years to fetch data for
    years = [2023, 2025]  # 2023-24 and 2024-25 seasons
    
    # Step 1: Create training dataset
    print("\nStep 1: Creating training dataset...")
    training_data = create_training_dataset(years, output_path="data/training_data.csv")
    
    if training_data.empty:
        print("Failed to create training dataset. Exiting.")
        return
    
    # Step 2: Initialize predictor and load data
    print("\nStep 2: Loading and preprocessing data...")
    predictor = NBAAwardsPredictor()
    predictor.data = training_data
    # Filter training data to only include players with sufficient playing time (MVP candidates)
    predictor.data = predictor.preprocess_data(predictor.data, filter_min_games=True, min_games=41, min_mp=20.0)
    
    # Step 3: Get feature columns
    predictor.feature_columns = get_feature_columns(predictor.data)
    print(f"\nUsing {len(predictor.feature_columns)} features:")
    print(f"  {', '.join(predictor.feature_columns[:10])}..." if len(predictor.feature_columns) > 10 
          else f"  {', '.join(predictor.feature_columns)}")
    
    # Step 4: Train models for each award
    print("\nStep 3: Training models for each award...")
    print("=" * 60)
    
    award_targets = {
        'MVP': 'won_MVP',
        'DPOY': 'won_DPOY',
        'ROY': 'won_ROY',
        '6MOY': 'won_6MOY',
        'MIP': 'won_MIP',
        'COY': 'won_COY'
    }
    
    all_metrics = {}
    
    for award_name, target_col in award_targets.items():
        if target_col in predictor.data.columns:
            # Check if we have any positive examples
            positive_count = predictor.data[target_col].sum()
            if positive_count > 0:
                print(f"\nTraining {award_name} model ({positive_count} winners)...")
                metrics = predictor.train_model(
                    award_name=award_name,
                    features=predictor.feature_columns,
                    target=target_col,
                    model_type='random_forest'
                )
                all_metrics[award_name] = metrics
            else:
                print(f"\n⚠ Skipping {award_name} - no winners in dataset")
        else:
            print(f"\n⚠ Skipping {award_name} - target column not found")
    
    # Step 5: Save models
    print("\nStep 4: Saving models...")
    predictor.save_models()
    
    # Step 6: Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModels trained: {len(predictor.models)}")
    print(f"Training data shape: {predictor.data.shape}")
    print(f"\nModel Performance Summary:")
    for award, metrics in all_metrics.items():
        print(f"  {award}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
    
    # Step 7: Example prediction (optional)
    print("\n" + "=" * 60)
    print("Example: Making predictions on training data...")
    print("=" * 60)
    
    # Get latest year data for example predictions
    latest_year_data = predictor.data[predictor.data['Year'] == max(years)].copy()
    if len(latest_year_data) > 0 and len(predictor.models) > 0:
        predictions = predictor.predict_all_awards(latest_year_data)
        print("\nTop 5 MVP Predictions:")
        # Show available probability columns
        prob_cols = [col for col in predictions.columns if col.endswith('_probability')]
        display_cols = ['Player'] + prob_cols[:3]  # Show first 3 award probabilities
        print(predictions[display_cols].head())


def predict_mvp_for_year(year: int, models_dir: str = "models", 
                         output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Predict MVP winner for a given year using advanced stats.
    
    Args:
        year: NBA season year to predict for (e.g., 2026)
        models_dir: Directory containing trained models
        output_path: Optional path to save predictions CSV
        
    Returns:
        DataFrame with players and their MVP probabilities, sorted by probability
    """
    print(f"\n{'='*60}")
    print(f"Predicting {year} MVP Winner")
    print(f"{'='*60}")
    
    # Load predictor with trained models
    predictor = NBAAwardsPredictor()
    predictor.load_models(models_dir)
    
    if 'MVP' not in predictor.models:
        raise ValueError("MVP model not found. Please train the model first by running main().")
    
    if predictor.feature_columns is None:
        raise ValueError("Feature columns not found. Please train the model first.")
    
    # Fetch advanced stats for the prediction year
    print(f"\nFetching {year} advanced stats...")
    adv_stats = fetch_advanced_stats(year)
    
    if adv_stats.empty:
        raise ValueError(f"Could not fetch advanced stats for {year}")
    
    # Filter players BEFORE making predictions - only predict on qualified candidates
    print(f"\nFiltering players (MVP candidates need PER>=15, G>=15, MP>=300)...")
    
    initial_count = len(adv_stats)
    
    # Convert to numeric for filtering
    for col in ['PER', 'G', 'MP', 'WS']:
        if col in adv_stats.columns:
            adv_stats[col] = pd.to_numeric(adv_stats[col], errors='coerce')
    
    # Filter BEFORE prediction - MVP candidates need:
    # - PER >= 15 (at least average performance)
    # - At least 15 games (reasonable sample)
    # - At least 300 total minutes (meaningful playing time)
    if 'PER' in adv_stats.columns:
        adv_stats = adv_stats[(adv_stats['PER'] >= 15.0) | (adv_stats['PER'].isna())]
    if 'G' in adv_stats.columns:
        adv_stats = adv_stats[(adv_stats['G'] >= 15.0) | (adv_stats['G'].isna())]
    if 'MP' in adv_stats.columns:
        adv_stats = adv_stats[(adv_stats['MP'] >= 300.0) | (adv_stats['MP'].isna())]
    
    filtered_count = len(adv_stats)
    if initial_count != filtered_count:
        print(f"  Filtered {initial_count - filtered_count} players, {filtered_count} qualified candidates remaining")
    
    # Now preprocess the filtered data
    adv_stats = predictor.preprocess_data(adv_stats, filter_min_games=False)
    
    # Make predictions only on qualified players
    print(f"\nMaking MVP predictions for {len(adv_stats)} qualified players...")
    predictions = predictor.predict_all_awards(adv_stats)
    
    # Sort by MVP probability
    if 'MVP_probability' in predictions.columns:
        predictions = predictions.sort_values('MVP_probability', ascending=False)
    
    # Merge with original stats to show context (use the filtered adv_stats)
    if 'Player' in adv_stats.columns:
        # Add key stats to predictions for context
        stats_to_show = ['Player']
        for col in ['G', 'GP', 'MP', 'PTS', 'TRB', 'AST', 'PER', 'WS']:
            if col in adv_stats.columns:
                stats_to_show.append(col)
        
        predictions = predictions.merge(
            adv_stats[stats_to_show], 
            on='Player', 
            how='left',
            suffixes=('', '_stats')
        )
    
    # Display top 10 predictions with context
    print(f"\n{'='*60}")
    print(f"Top 10 {year} MVP Predictions:")
    print(f"{'='*60}")
    
    # Show relevant columns
    display_cols = ['Player', 'MVP_probability']
    for col in ['G', 'GP', 'MP', 'PTS', 'PER']:
        if col in predictions.columns:
            display_cols.append(col)
    
    print(predictions[display_cols].head(10).to_string(index=False))
    
    # Save predictions if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
    
    return predictions


def main():
    """Main function."""
    print("NBA Awards Predictor")
    print("=" * 60)
    
    # Years to fetch data for
    years = [2023, 2025]  # 2023-24 and 2024-25 seasons
    
    # Step 1: Create training dataset
    print("\nStep 1: Creating training dataset...")
    training_data = create_training_dataset(years, output_path="data/training_data.csv")
    
    if training_data.empty:
        print("Failed to create training dataset. Exiting.")
        return
    
    # Step 2: Initialize predictor and load data
    print("\nStep 2: Loading and preprocessing data...")
    predictor = NBAAwardsPredictor()
    predictor.data = training_data
    # Filter training data to only include players with sufficient playing time (MVP candidates)
    predictor.data = predictor.preprocess_data(predictor.data, filter_min_games=True, min_games=41, min_mp=20.0)
    
    # Step 3: Get feature columns
    predictor.feature_columns = get_feature_columns(predictor.data)
    print(f"\nUsing {len(predictor.feature_columns)} features:")
    print(f"  {', '.join(predictor.feature_columns[:10])}..." if len(predictor.feature_columns) > 10 
          else f"  {', '.join(predictor.feature_columns)}")
    
    # Step 4: Train models for each award
    print("\nStep 3: Training models for each award...")
    print("=" * 60)
    
    award_targets = {
        'MVP': 'won_MVP',
        'DPOY': 'won_DPOY',
        'ROY': 'won_ROY',
        '6MOY': 'won_6MOY',
        'MIP': 'won_MIP',
        'COY': 'won_COY'
    }
    
    all_metrics = {}
    
    for award_name, target_col in award_targets.items():
        if target_col in predictor.data.columns:
            # Check if we have any positive examples
            positive_count = predictor.data[target_col].sum()
            if positive_count > 0:
                print(f"\nTraining {award_name} model ({positive_count} winners)...")
                metrics = predictor.train_model(
                    award_name=award_name,
                    features=predictor.feature_columns,
                    target=target_col,
                    model_type='random_forest'
                )
                all_metrics[award_name] = metrics
            else:
                print(f"\n⚠ Skipping {award_name} - no winners in dataset")
        else:
            print(f"\n⚠ Skipping {award_name} - target column not found")
    
    # Step 5: Save models
    print("\nStep 4: Saving models...")
    predictor.save_models()
    
    # Step 6: Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModels trained: {len(predictor.models)}")
    print(f"Training data shape: {predictor.data.shape}")
    print(f"\nModel Performance Summary:")
    for award, metrics in all_metrics.items():
        print(f"  {award}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
    
    # Step 7: Example prediction (optional)
    print("\n" + "=" * 60)
    print("Example: Making predictions on training data...")
    print("=" * 60)
    
    # Get latest year data for example predictions
    latest_year_data = predictor.data[predictor.data['Year'] == max(years)].copy()
    if len(latest_year_data) > 0 and len(predictor.models) > 0:
        predictions = predictor.predict_all_awards(latest_year_data)
        print("\nTop 5 MVP Predictions:")
        # Show available probability columns
        prob_cols = [col for col in predictions.columns if col.endswith('_probability')]
        display_cols = ['Player'] + prob_cols[:3]  # Show first 3 award probabilities
        print(predictions[display_cols].head())


if __name__ == "__main__":
    import sys
    
    # Check if user wants to predict for a specific year
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        year = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
        try:
            predict_mvp_for_year(year, output_path=f"data/mvp_predictions_{year}.csv")
        except Exception as e:
            print(f"Error making prediction: {e}")
            print("\nMake sure you've trained the models first by running: python src/main.py")
    else:
        # Default: train models
        main()
