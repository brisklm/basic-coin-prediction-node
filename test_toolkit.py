"""
Test script for AI Competition Toolkit
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from ai_competition_toolkit import CompetitionFramework, quick_train, create_default_config

def test_basic_functionality():
    """Test basic functionality with small dataset"""
    print("Testing basic functionality...")
    
    # Create small test dataset
    X, y = make_classification(
        n_samples=100, 
        n_features=10, 
        n_informative=5,
        n_classes=2,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    # Test quick_train
    try:
        framework = CompetitionFramework()
        framework.config.set('max_trials', 5)  # Very few trials for testing
        framework.config.set('cv_folds', 2)
        
        X_train, y_train, _ = framework.prepare_data(df, 'target')
        framework.train_models(X_train, y_train)
        
        print("✓ Basic functionality test passed")
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_config_creation():
    """Test configuration file creation"""
    print("Testing configuration creation...")
    
    try:
        create_default_config('test_config.yaml')
        
        # Test loading config
        framework = CompetitionFramework('test_config.yaml')
        
        print("✓ Configuration test passed")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing pipeline"""
    print("Testing data preprocessing...")
    
    try:
        # Create test data with missing values and categorical variables
        np.random.seed(42)
        
        data = {
            'numeric1': np.random.randn(50),
            'numeric2': np.random.randn(50),
            'category': np.random.choice(['A', 'B', 'C'], 50),
            'target': np.random.randint(0, 2, 50)
        }
        
        df = pd.DataFrame(data)
        
        # Introduce missing values
        df.loc[0:5, 'numeric1'] = np.nan
        df.loc[10:15, 'category'] = np.nan
        
        framework = CompetitionFramework()
        X_train, y_train, _ = framework.prepare_data(df, 'target')
        
        # Check if missing values are handled
        assert not X_train.isnull().any().any(), "Missing values not handled"
        
        print("✓ Data preprocessing test passed")
        return True
    except Exception as e:
        print(f"✗ Data preprocessing test failed: {e}")
        return False

def test_model_training():
    """Test model training with minimal configuration"""
    print("Testing model training...")
    
    try:
        X, y = make_classification(
            n_samples=50, 
            n_features=5, 
            n_classes=2,
            random_state=42
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y
        
        framework = CompetitionFramework()
        framework.config.set('max_trials', 3)
        framework.config.set('cv_folds', 2)
        framework.config.set('models', {'lgb': True, 'rf': True})  # Only test 2 models
        
        X_train, y_train, _ = framework.prepare_data(df, 'target')
        models = framework.train_models(X_train, y_train)
        
        assert len(models) > 0, "No models trained"
        
        print("✓ Model training test passed")
        return True
    except Exception as e:
        print(f"✗ Model training test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*50)
    print("AI Competition Toolkit - Test Suite")
    print("="*50)
    
    tests = [
        test_config_creation,
        test_data_preprocessing,
        test_basic_functionality,
        test_model_training
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("="*50)
    
    if passed == total:
        print("🎉 All tests passed! The toolkit is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()