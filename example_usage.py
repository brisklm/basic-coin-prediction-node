"""
AI Competition Toolkit - Example Usage
=====================================

This file demonstrates how to use the AI Competition Toolkit for various scenarios.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from ai_competition_toolkit import CompetitionFramework, quick_train, create_default_config

def example_1_classification():
    """Example 1: Binary Classification Problem"""
    print("="*50)
    print("EXAMPLE 1: Binary Classification")
    print("="*50)
    
    # Generate sample classification data
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15, 
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=y)
    
    # Quick training approach
    print("Using quick_train() function...")
    framework = quick_train(train_df, 'target')
    
    # Evaluate on test set
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    # Process test data through the same pipeline
    X_test_processed = framework.preprocessor.transform(X_test)
    X_test_engineered = framework.feature_engineer.engineer_features(X_test_processed)
    
    # Ensure same features as training
    for col in framework.feature_engineer.feature_selector.get_feature_names_out():
        if col not in X_test_engineered.columns:
            X_test_engineered[col] = 0
    
    selected_features = framework.feature_engineer.feature_selector.get_feature_names_out()
    X_test_final = X_test_engineered[selected_features]
    
    # Evaluate models
    results = framework.evaluate_models(X_test_final, y_test)
    
    print(f"\nBest model: {max(results.keys(), key=lambda k: results[k])}")
    print(f"Best score: {max(results.values()):.4f}")
    
    # Save the trained framework
    framework.save_model('classification_model')
    print("Model saved successfully!")

def example_2_regression():
    """Example 2: Regression Problem"""
    print("\n" + "="*50)
    print("EXAMPLE 2: Regression")
    print("="*50)
    
    # Generate sample regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        noise=0.1,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Initialize framework with custom configuration
    framework = CompetitionFramework()
    
    # Customize configuration
    framework.config.set('max_trials', 50)  # Reduce trials for faster execution
    framework.config.set('models', {
        'lgb': True,
        'xgb': True,
        'rf': True,
        'lr': True,
        'catboost': False,  # Disable for faster execution
        'svm': False
    })
    
    # Prepare data
    X_train, y_train, X_test = framework.prepare_data(train_df, 'target', test_df)
    
    # Train models
    trained_models = framework.train_models(X_train, y_train)
    
    # Create ensembles
    framework.create_ensembles(X_train, y_train)
    
    # Evaluate
    y_test = test_df['target']
    results = framework.evaluate_models(X_test, y_test)
    
    print(f"\nBest model: {max(results.keys(), key=lambda k: results[k])}")
    print(f"Best score (negative MSE): {max(results.values()):.4f}")

def example_3_custom_config():
    """Example 3: Using Custom Configuration"""
    print("\n" + "="*50)
    print("EXAMPLE 3: Custom Configuration")
    print("="*50)
    
    # Create a custom configuration file
    create_default_config('my_config.yaml')
    
    # Load and modify configuration
    framework = CompetitionFramework('my_config.yaml')
    
    # Customize settings
    framework.config.set('cv_folds', 3)
    framework.config.set('max_trials', 30)
    framework.config.set('ensemble_methods', ['voting'])
    framework.config.set('preprocessing', {
        'handle_missing': True,
        'encode_categorical': True,
        'scale_features': True,
        'remove_outliers': True  # Enable outlier removal
    })
    
    # Save modified configuration
    framework.config.save_config('my_modified_config.yaml')
    
    print("Custom configuration created and saved!")
    print("Configuration settings:")
    for key, value in framework.config.config.items():
        print(f"  {key}: {value}")

def example_4_competition_simulation():
    """Example 4: Complete Competition Simulation"""
    print("\n" + "="*50)
    print("EXAMPLE 4: Competition Simulation")
    print("="*50)
    
    # Simulate competition data with missing values and mixed types
    np.random.seed(42)
    
    # Create mixed dataset
    n_samples = 800
    
    # Numerical features
    numerical_data = np.random.randn(n_samples, 8)
    
    # Categorical features
    categories = ['A', 'B', 'C', 'D']
    categorical_data = np.random.choice(categories, size=(n_samples, 3))
    
    # Create target (classification)
    target = (numerical_data[:, 0] + numerical_data[:, 1] > 0).astype(int)
    
    # Add some noise
    noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    target[noise_idx] = 1 - target[noise_idx]
    
    # Create DataFrame
    df = pd.DataFrame(numerical_data, columns=[f'num_{i}' for i in range(8)])
    for i, col in enumerate([f'cat_{j}' for j in range(3)]):
        df[col] = categorical_data[:, i]
    
    df['target'] = target
    
    # Introduce missing values
    missing_mask = np.random.random(df.shape) < 0.05
    df = df.mask(missing_mask)
    
    # Split into train and test
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    # Remove target from test set (simulate competition format)
    test_features = test_df.drop('target', axis=1)
    test_target = test_df['target']
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_features.shape}")
    print(f"Missing values in training: {train_df.isnull().sum().sum()}")
    
    # Train using the framework
    framework = CompetitionFramework()
    
    # Set configuration for robust handling
    framework.config.set('preprocessing', {
        'handle_missing': True,
        'encode_categorical': True,
        'scale_features': True,
        'remove_outliers': False
    })
    framework.config.set('max_trials', 20)  # Faster for demo
    
    # Prepare and train
    X_train, y_train, X_test = framework.prepare_data(train_df, 'target', test_features)
    
    # Train models
    framework.train_models(X_train, y_train)
    
    # Create ensembles
    framework.create_ensembles(X_train, y_train)
    
    # Create submission format
    submission_format = pd.DataFrame({
        'id': range(len(test_features)),
        'prediction': 0  # Placeholder
    })
    
    # Generate submission file
    submission = framework.generate_submission(
        X_test, 
        submission_format, 
        'competition_submission.csv',
        use_ensemble='stacking'
    )
    
    # Evaluate performance (this wouldn't be available in real competition)
    results = framework.evaluate_models(X_test, test_target)
    print(f"\nActual performance on hidden test set:")
    for model, score in results.items():
        print(f"{model}: {score:.4f}")

def example_5_model_comparison():
    """Example 5: Detailed Model Comparison"""
    print("\n" + "="*50)
    print("EXAMPLE 5: Model Comparison")
    print("="*50)
    
    # Create a challenging dataset
    X, y = make_classification(
        n_samples=500,
        n_features=30,
        n_informative=15,
        n_redundant=10,
        n_clusters_per_class=1,
        class_sep=0.8,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=y)
    
    # Test different ensemble methods
    ensemble_methods = ['voting', 'stacking']
    
    for method in ensemble_methods:
        print(f"\n--- Testing {method.upper()} ensemble ---")
        
        framework = CompetitionFramework()
        framework.config.set('ensemble_methods', [method])
        framework.config.set('max_trials', 15)
        
        # Train
        X_train, y_train, X_test = framework.prepare_data(train_df, 'target', test_df)
        framework.train_models(X_train, y_train)
        framework.create_ensembles(X_train, y_train)
        
        # Evaluate
        y_test = test_df['target']
        results = framework.evaluate_models(X_test, y_test)
        
        print(f"Best individual model: {max([k for k in results.keys() if not k.startswith('ensemble')], key=lambda k: results[k])}")
        if f'ensemble_{method}' in results:
            print(f"Ensemble performance: {results[f'ensemble_{method}']:.4f}")

if __name__ == "__main__":
    # Run all examples
    print("AI Competition Toolkit - Examples")
    print("=================================")
    
    try:
        example_1_classification()
        example_2_regression()
        example_3_custom_config()
        example_4_competition_simulation()
        example_5_model_comparison()
        
        print("\n" + "="*50)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()