"""
AI Competition Toolkit - A general-purpose framework for machine learning competitions
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (accuracy_score, roc_auc_score, mean_squared_error, 
                           mean_absolute_error, log_loss, f1_score)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from typing import Dict, List, Any, Optional, Tuple, Union
import yaml
import joblib
import warnings
warnings.filterwarnings('ignore')

class CompetitionConfig:
    """Configuration management for competition parameters"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.default_config = {
            'problem_type': 'auto',  # 'classification', 'regression', 'auto'
            'target_column': 'target',
            'metric': 'auto',  # Will be auto-detected based on problem type
            'cv_folds': 5,
            'random_state': 42,
            'test_size': 0.2,
            'max_trials': 100,
            'feature_selection': True,
            'feature_engineering': True,
            'ensemble_methods': ['voting', 'stacking'],
            'models': {
                'lgb': True,
                'xgb': True,
                'catboost': True,
                'rf': True,
                'lr': True,
                'svm': False  # Disabled by default for large datasets
            },
            'preprocessing': {
                'handle_missing': True,
                'encode_categorical': True,
                'scale_features': True,
                'remove_outliers': False
            }
        }
        
        if config_file:
            self.load_config(config_file)
        else:
            self.config = self.default_config.copy()
    
    def load_config(self, config_file: str):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                custom_config = yaml.safe_load(f)
            self.config = {**self.default_config, **custom_config}
        except FileNotFoundError:
            print(f"Config file {config_file} not found. Using default configuration.")
            self.config = self.default_config.copy()
    
    def save_config(self, config_file: str):
        """Save current configuration to YAML file"""
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value

class DataPreprocessor:
    """Comprehensive data preprocessing pipeline"""
    
    def __init__(self, config: CompetitionConfig):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.is_fitted = False
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Fit preprocessor and transform data; returns (X_processed, y_processed)."""
        X_processed = X.copy()
        
        # Handle missing values
        if self.config.get('preprocessing')['handle_missing']:
            X_processed = self._handle_missing_values(X_processed)
        
        # Encode categorical variables
        if self.config.get('preprocessing')['encode_categorical']:
            X_processed = self._encode_categorical(X_processed, fit=True)
        
        # Remove outliers (optional)
        if self.config.get('preprocessing')['remove_outliers'] and y is not None:
            X_processed, y = self._remove_outliers(X_processed, y)
        
        # Scale features
        if self.config.get('preprocessing')['scale_features']:
            X_processed = self._scale_features(X_processed, fit=True)
        
        self.feature_names = X_processed.columns.tolist()
        self.is_fitted = True
        
        return X_processed, y
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X_processed = X.copy()
        
        # Handle missing values
        if self.config.get('preprocessing')['handle_missing']:
            X_processed = self._handle_missing_values(X_processed)
        
        # Encode categorical variables
        if self.config.get('preprocessing')['encode_categorical']:
            X_processed = self._encode_categorical(X_processed, fit=False)
        
        # Scale features
        if self.config.get('preprocessing')['scale_features']:
            X_processed = self._scale_features(X_processed, fit=False)
        
        # Ensure same columns as training
        for col in self.feature_names:
            if col not in X_processed.columns:
                X_processed[col] = 0
        
        X_processed = X_processed[self.feature_names]
        
        return X_processed
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies"""
        X_filled = X.copy()
        
        for col in X_filled.columns:
            if X_filled[col].dtype in ['object', 'category']:
                # Fill categorical with mode or 'unknown'
                mode_val = X_filled[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'unknown'
                X_filled[col] = X_filled[col].fillna(fill_val)
            else:
                # Fill numerical with median
                X_filled[col] = X_filled[col].fillna(X_filled[col].median())
        
        return X_filled
    
    def _encode_categorical(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical variables"""
        X_encoded = X.copy()
        
        categorical_cols = X_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if fit:
                # Use frequency encoding for high cardinality, one-hot for low cardinality
                unique_vals = X_encoded[col].nunique()
                if unique_vals > 10:
                    # Frequency encoding
                    freq_map = X_encoded[col].value_counts().to_dict()
                    self.encoders[col] = ('frequency', freq_map)
                    X_encoded[col] = X_encoded[col].map(freq_map).fillna(0)
                else:
                    # One-hot encoding
                    dummies = pd.get_dummies(X_encoded[col], prefix=col)
                    self.encoders[col] = ('onehot', dummies.columns.tolist())
                    X_encoded = pd.concat([X_encoded.drop(col, axis=1), dummies], axis=1)
            else:
                # Apply fitted encoding
                if col in self.encoders:
                    encoding_type, encoding_data = self.encoders[col]
                    if encoding_type == 'frequency':
                        X_encoded[col] = X_encoded[col].map(encoding_data).fillna(0)
                    elif encoding_type == 'onehot':
                        dummies = pd.get_dummies(X_encoded[col], prefix=col)
                        # Ensure all columns exist
                        for dummy_col in encoding_data:
                            if dummy_col not in dummies.columns:
                                dummies[dummy_col] = 0
                        dummies = dummies[encoding_data]
                        X_encoded = pd.concat([X_encoded.drop(col, axis=1), dummies], axis=1)
        
        return X_encoded
    
    def _scale_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale numerical features"""
        X_scaled = X.copy()
        numerical_cols = X_scaled.select_dtypes(include=[np.number, 'bool']).columns
        
        if fit:
            self.scalers['scaler'] = RobustScaler()
            X_scaled[numerical_cols] = self.scalers['scaler'].fit_transform(X_scaled[numerical_cols])
        else:
            if 'scaler' in self.scalers:
                X_scaled[numerical_cols] = self.scalers['scaler'].transform(X_scaled[numerical_cols])
        
        return X_scaled
    
    def _remove_outliers(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Remove outliers using IQR method"""
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers
        outlier_mask = ((X < lower_bound) | (X > upper_bound)).any(axis=1)
        
        # Remove outliers
        X_clean = X[~outlier_mask]
        y_clean = y[~outlier_mask]
        
        print(f"Removed {outlier_mask.sum()} outliers ({outlier_mask.mean():.2%} of data)")
        
        return X_clean, y_clean

class FeatureEngineer:
    """Advanced feature engineering pipeline"""
    
    def __init__(self, config: CompetitionConfig):
        self.config = config
        self.feature_selector = None
        self.generated_features = []
    
    def engineer_features(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Generate new features from existing ones"""
        if not self.config.get('feature_engineering'):
            return X
        
        X_engineered = X.copy()
        
        # Generate polynomial features for numerical columns
        numerical_cols = X_engineered.select_dtypes(include=[np.number]).columns[:10]  # Limit to avoid explosion
        
        if len(numerical_cols) > 1:
            # Create interaction features
            for i, col1 in enumerate(numerical_cols):
                for col2 in numerical_cols[i+1:]:
                    new_feature = f"{col1}_x_{col2}"
                    X_engineered[new_feature] = X_engineered[col1] * X_engineered[col2]
                    self.generated_features.append(new_feature)
                
                # Create squared features
                squared_feature = f"{col1}_squared"
                X_engineered[squared_feature] = X_engineered[col1] ** 2
                self.generated_features.append(squared_feature)
        
        # Generate statistical features
        if len(numerical_cols) > 2:
            X_engineered['mean_all'] = X_engineered[numerical_cols].mean(axis=1)
            X_engineered['std_all'] = X_engineered[numerical_cols].std(axis=1)
            X_engineered['min_all'] = X_engineered[numerical_cols].min(axis=1)
            X_engineered['max_all'] = X_engineered[numerical_cols].max(axis=1)
            
            self.generated_features.extend(['mean_all', 'std_all', 'min_all', 'max_all'])
        
        return X_engineered
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> pd.DataFrame:
        """Select best features using statistical tests"""
        if not self.config.get('feature_selection'):
            return X
        
        # Determine number of features to select
        n_features = min(X.shape[1], max(10, int(X.shape[1] * 0.8)))
        
        if problem_type == 'classification':
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        else:
            self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
        
        X_selected = self.feature_selector.fit_transform(X, y)
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        print(f"Selected {len(selected_features)} features out of {X.shape[1]}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

class ModelOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, config: CompetitionConfig):
        self.config = config
        self.best_params = {}
        self.study_results = {}
    
    def optimize_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, 
                      problem_type: str, metric_func) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific model"""
        
        def objective(trial):
            params = self._get_trial_params(trial, model_name, problem_type)
            model = self._create_model(model_name, params, problem_type)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=self.config.get('cv_folds'), 
                               shuffle=True, random_state=self.config.get('random_state'))
            if problem_type == 'regression':
                cv = KFold(n_splits=self.config.get('cv_folds'), 
                         shuffle=True, random_state=self.config.get('random_state'))
            
            # Ensure X is numeric for models that don't handle objects directly (e.g., xgboost)
            X_for_cv = X
            try:
                if model_name == 'xgb' and any(getattr(X_for_cv[c], 'dtype', None) == 'object' for c in X_for_cv.columns):
                    import pandas as pd  # local import for safety
                    X_for_cv = pd.get_dummies(X_for_cv, drop_first=True)
            except Exception:
                pass

            try:
                scores = cross_val_score(model, X_for_cv, y, cv=cv, scoring=metric_func, n_jobs=-1, error_score=0.0)
                return scores.mean()
            except Exception:
                # Treat invalid configurations as zero score to keep optimization running
                return 0.0
        
        study = optuna.create_study(direction='maximize' if problem_type == 'classification' else 'minimize')
        study.optimize(objective, n_trials=self.config.get('max_trials'), show_progress_bar=True)
        
        self.best_params[model_name] = study.best_params
        self.study_results[model_name] = {
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials)
        }
        
        return study.best_params
    
    def _get_trial_params(self, trial, model_name: str, problem_type: str) -> Dict[str, Any]:
        """Get hyperparameter suggestions for each model"""
        
        if model_name == 'lgb':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            }
        
        elif model_name == 'xgb':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            }
        
        elif model_name == 'catboost':
            return {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 30),
                'border_count': trial.suggest_int('border_count', 32, 255),
            }
        
        elif model_name == 'rf':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            }
        
        elif model_name == 'lr':
            if problem_type == 'classification':
                C_val = trial.suggest_float('C', 0.01, 100, log=True)
                penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
                # Choose a compatible solver for the chosen penalty
                if penalty == 'elasticnet':
                    solver = 'saga'
                else:
                    solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
                    # liblinear does not support elasticnet; ensure compatibility
                    if solver == 'liblinear' and penalty == 'l2':
                        pass  # ok
                    elif solver == 'liblinear' and penalty == 'l1':
                        pass  # ok
                    elif solver == 'liblinear':
                        solver = 'saga'

                params = {
                    'C': C_val,
                    'penalty': penalty,
                    'solver': solver,
                    # Handle class imbalance more robustly
                    'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
                }
                if penalty == 'elasticnet':
                    params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
                return params
            else:
                return {
                    'alpha': trial.suggest_float('alpha', 0.01, 100, log=True),
                }
        
        return {}
    
    def _create_model(self, model_name: str, params: Dict[str, Any], problem_type: str):
        """Create model instance with given parameters"""
        
        base_params = {'random_state': self.config.get('random_state')}
        params = {**base_params, **params}
        
        if model_name == 'lgb':
            if problem_type == 'classification':
                params['objective'] = 'binary'
                return lgb.LGBMClassifier(**params)
            else:
                params['objective'] = 'regression'
                return lgb.LGBMRegressor(**params)
        
        elif model_name == 'xgb':
            if problem_type == 'classification':
                # Enable safer defaults to reduce dtype issues
                params.setdefault('enable_categorical', True)
                params.setdefault('use_label_encoder', False)
                return xgb.XGBClassifier(**params)
            else:
                params.setdefault('enable_categorical', True)
                return xgb.XGBRegressor(**params)
        
        elif model_name == 'catboost':
            params['verbose'] = False
            if problem_type == 'classification':
                return cb.CatBoostClassifier(**params)
            else:
                return cb.CatBoostRegressor(**params)
        
        elif model_name == 'rf':
            if problem_type == 'classification':
                return RandomForestClassifier(**params)
            else:
                return RandomForestRegressor(**params)
        
        elif model_name == 'lr':
            if problem_type == 'classification':
                return LogisticRegression(**params)
            else:
                return Ridge(**params)
        
        return None

class EnsembleManager:
    """Ensemble methods for combining multiple models"""
    
    def __init__(self, config: CompetitionConfig):
        self.config = config
        self.models = {}
        self.ensemble_models = {}
        self.model_weights = {}
    
    def add_model(self, name: str, model, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.model_weights[name] = weight
    
    def create_voting_ensemble(self, X: pd.DataFrame, y: pd.Series, problem_type: str):
        """Create voting ensemble"""
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        
        model_list = [(name, model) for name, model in self.models.items()]
        
        if problem_type == 'classification':
            ensemble = VotingClassifier(estimators=model_list, voting='soft')
        else:
            ensemble = VotingRegressor(estimators=model_list)
        
        ensemble.fit(X, y)
        self.ensemble_models['voting'] = ensemble
        return ensemble
    
    def create_stacking_ensemble(self, X: pd.DataFrame, y: pd.Series, problem_type: str):
        """Create stacking ensemble"""
        from sklearn.ensemble import StackingClassifier, StackingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        
        model_list = [(name, model) for name, model in self.models.items()]
        
        if problem_type == 'classification':
            meta_model = LogisticRegression(random_state=self.config.get('random_state'))
            ensemble = StackingClassifier(
                estimators=model_list, 
                final_estimator=meta_model,
                cv=self.config.get('cv_folds')
            )
        else:
            meta_model = Ridge(random_state=self.config.get('random_state'))
            ensemble = StackingRegressor(
                estimators=model_list,
                final_estimator=meta_model,
                cv=self.config.get('cv_folds')
            )
        
        ensemble.fit(X, y)
        self.ensemble_models['stacking'] = ensemble
        return ensemble
    
    def create_weighted_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create weighted average of predictions"""
        weighted_pred = np.zeros_like(list(predictions.values())[0])
        total_weight = sum(self.model_weights.values())
        
        for name, pred in predictions.items():
            weight = self.model_weights.get(name, 1.0)
            weighted_pred += (weight / total_weight) * pred
        
        return weighted_pred

class CompetitionFramework:
    """Main framework that orchestrates the entire competition pipeline"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = CompetitionConfig(config_file)
        self.preprocessor = DataPreprocessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_optimizer = ModelOptimizer(self.config)
        self.ensemble_manager = EnsembleManager(self.config)
        
        self.problem_type = None
        self.metric = None
        self.trained_models = {}
        self.results = {}
        
    def detect_problem_type(self, y: pd.Series) -> str:
        """Auto-detect problem type based on target variable"""
        if self.config.get('problem_type') != 'auto':
            return self.config.get('problem_type')
        
        unique_values = y.nunique()
        
        if y.dtype == 'object' or unique_values <= 20:
            return 'classification'
        else:
            return 'regression'
    
    def get_metric(self, problem_type: str) -> str:
        """Get appropriate metric for the problem type"""
        if self.config.get('metric') != 'auto':
            return self.config.get('metric')
        
        if problem_type == 'classification':
            # Use y_train if available, otherwise default to binary classification
            if hasattr(self, 'y_train') and self.y_train is not None:
                return 'roc_auc' if len(np.unique(self.y_train)) == 2 else 'accuracy'
            else:
                return 'roc_auc'  # Default for binary classification
        else:
            return 'neg_mean_squared_error'
    
    def prepare_data(self, train_data: pd.DataFrame, 
                    target_column: str = None,
                    test_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
        """Prepare training and test data"""
        
        target_col = target_column or self.config.get('target_column')

        # Robust target column detection if configured target is missing
        if target_col not in train_data.columns:
            inferred_target: Optional[str] = None

            # 1) If test data provided, infer by column difference (train has target, test doesn't)
            if test_data is not None:
                try:
                    diff_cols = [c for c in train_data.columns if c not in test_data.columns]
                    # Filter out common id-like columns heuristically if needed later
                    if len(diff_cols) == 1:
                        inferred_target = diff_cols[0]
                except Exception:
                    pass

            # 2) Try common target names
            if inferred_target is None:
                common_targets = [
                    'target', 'Target', 'survived', 'Survived', 'SalePrice', 'saleprice',
                    'label', 'Label', 'y'
                ]
                for candidate in common_targets:
                    if candidate in train_data.columns:
                        inferred_target = candidate
                        break

            # 3) Fallback to the last column in train if still unknown
            if inferred_target is None and len(train_data.columns) > 0:
                inferred_target = train_data.columns[-1]

            if inferred_target is None:
                raise KeyError(f"Could not determine target column. Provided '{target_col}' not found and auto-detection failed.")

            print(f"Warning: target column '{target_col}' not found. Auto-detected '{inferred_target}' and will use it.")
            target_col = inferred_target
            # Persist chosen target for downstream steps
            self.config.set('target_column', target_col)
        
        # Split features and target
        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]
        
        # Detect problem type
        self.problem_type = self.detect_problem_type(y_train)
        self.metric = self.get_metric(self.problem_type)
        
        print(f"Detected problem type: {self.problem_type}")
        print(f"Using metric: {self.metric}")
        
        # Store original data
        self.y_train = y_train
        
        # Preprocess data
        X_train_processed, y_train_processed = self.preprocessor.fit_transform(X_train, y_train)
        
        # Feature engineering
        X_train_engineered = self.feature_engineer.engineer_features(X_train_processed, y_train_processed)
        
        # Feature selection
        X_train_final = self.feature_engineer.select_features(X_train_engineered, y_train_processed, self.problem_type)
        
        # Process test data if provided
        X_test_final = None
        if test_data is not None:
            X_test = test_data.copy()
            if target_col in X_test.columns:
                X_test = X_test.drop(columns=[target_col])
            
            X_test_processed = self.preprocessor.transform(X_test)
            X_test_engineered = self.feature_engineer.engineer_features(X_test_processed)
            
            # Ensure test data has same features as training data
            for col in X_train_final.columns:
                if col not in X_test_engineered.columns:
                    X_test_engineered[col] = 0
            
            X_test_final = X_test_engineered[X_train_final.columns]
        
        return X_train_final, y_train_processed, X_test_final
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train all enabled models with hyperparameter optimization"""
        
        print("Starting model training and optimization...")
        
        enabled_models = {name: enabled for name, enabled in self.config.get('models').items() if enabled}
        
        for model_name in enabled_models:
            print(f"\nOptimizing {model_name.upper()}...")
            
            # Optimize hyperparameters
            best_params = self.model_optimizer.optimize_model(
                model_name, X_train, y_train, self.problem_type, self.metric
            )
            
            # Train final model with best parameters
            final_model = self.model_optimizer._create_model(model_name, best_params, self.problem_type)
            final_model.fit(X_train, y_train)
            
            self.trained_models[model_name] = final_model
            self.ensemble_manager.add_model(model_name, final_model)
            
            print(f"{model_name.upper()} optimization completed")
        # Fallback: ensure at least one baseline model is trained
        if not self.trained_models:
            print("No models were trained via optimization; fitting a baseline model.")
            if self.problem_type == 'regression':
                baseline = Ridge()
            else:
                baseline = LogisticRegression(solver='saga', max_iter=1000)
            baseline.fit(X_train, y_train)
            self.trained_models['baseline'] = baseline
            self.ensemble_manager.add_model('baseline', baseline)
            print("Baseline model trained.")

        return self.trained_models
    
    def create_ensembles(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Create ensemble models"""
        
        print("\nCreating ensemble models...")
        
        ensemble_methods = self.config.get('ensemble_methods', [])
        
        if 'voting' in ensemble_methods and len(self.trained_models) > 1:
            print("Creating voting ensemble...")
            self.ensemble_manager.create_voting_ensemble(X_train, y_train, self.problem_type)
        
        if 'stacking' in ensemble_methods and len(self.trained_models) > 1:
            print("Creating stacking ensemble...")
            self.ensemble_manager.create_stacking_ensemble(X_train, y_train, self.problem_type)
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate all models on test data"""
        
        results = {}
        
        # Evaluate individual models
        for name, model in self.trained_models.items():
            predictions = model.predict(X_test)
            score = self._calculate_score(y_test, predictions)
            results[name] = score
            print(f"{name.upper()}: {score:.4f}")
        
        # Evaluate ensemble models
        for name, ensemble in self.ensemble_manager.ensemble_models.items():
            predictions = ensemble.predict(X_test)
            score = self._calculate_score(y_test, predictions)
            results[f"ensemble_{name}"] = score
            print(f"ENSEMBLE {name.upper()}: {score:.4f}")
        
        self.results = results
        return results
    
    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate score based on problem type and metric"""
        
        if self.problem_type == 'classification':
            if self.metric == 'roc_auc':
                return roc_auc_score(y_true, y_pred)
            elif self.metric == 'accuracy':
                return accuracy_score(y_true, y_pred)
            elif self.metric == 'f1':
                return f1_score(y_true, y_pred, average='weighted')
        else:
            if self.metric == 'neg_mean_squared_error':
                return -mean_squared_error(y_true, y_pred)
            elif self.metric == 'neg_mean_absolute_error':
                return -mean_absolute_error(y_true, y_pred)
        
        return 0.0
    
    def predict(self, X_test: pd.DataFrame, use_ensemble: str = 'stacking') -> np.ndarray:
        """Make predictions on test data"""
        
        # Use ensemble if available
        if use_ensemble in self.ensemble_manager.ensemble_models:
            return self.ensemble_manager.ensemble_models[use_ensemble].predict(X_test)
        
        # Otherwise use best individual model
        if self.results:
            best_model_name = max(self.results.keys(), key=lambda k: self.results[k])
            if best_model_name in self.trained_models:
                return self.trained_models[best_model_name].predict(X_test)
        
        # Fallback to first available model
        if self.trained_models:
            first_model = list(self.trained_models.values())[0]
            return first_model.predict(X_test)
        
        raise ValueError("No trained models available for prediction")
    
    def save_model(self, filepath: str, model_name: Optional[str] = None):
        """Save trained model(s) to file"""
        
        if model_name:
            if model_name in self.trained_models:
                joblib.dump(self.trained_models[model_name], f"{filepath}_{model_name}.pkl")
            elif f"ensemble_{model_name}" in self.ensemble_manager.ensemble_models:
                joblib.dump(self.ensemble_manager.ensemble_models[f"ensemble_{model_name}"], f"{filepath}_ensemble_{model_name}.pkl")
        else:
            # Save all models
            save_data = {
                'models': self.trained_models,
                'ensembles': self.ensemble_manager.ensemble_models,
                'preprocessor': self.preprocessor,
                'feature_engineer': self.feature_engineer,
                'config': self.config.config,
                'problem_type': self.problem_type,
                'results': self.results
            }
            joblib.dump(save_data, f"{filepath}_complete.pkl")
    
    def load_model(self, filepath: str):
        """Load trained model(s) from file"""
        
        loaded_data = joblib.load(filepath)
        
        if isinstance(loaded_data, dict) and 'models' in loaded_data:
            # Load complete framework
            self.trained_models = loaded_data['models']
            self.ensemble_manager.ensemble_models = loaded_data['ensembles']
            self.preprocessor = loaded_data['preprocessor']
            self.feature_engineer = loaded_data['feature_engineer']
            self.config.config = loaded_data['config']
            self.problem_type = loaded_data['problem_type']
            self.results = loaded_data.get('results', {})
        else:
            # Load single model
            return loaded_data
    
    def generate_submission(self, X_test: pd.DataFrame, 
                          submission_format: pd.DataFrame,
                          output_file: str = 'submission.csv',
                          use_ensemble: str = 'stacking'):
        """Generate submission file for competition"""
        
        predictions = self.predict(X_test, use_ensemble)
        
        submission = submission_format.copy()
        target_col = submission.columns[-1]  # Assume last column is target
        submission[target_col] = predictions
        
        submission.to_csv(output_file, index=False)
        print(f"Submission saved to {output_file}")
        
        return submission

# Utility functions for quick usage

def quick_train(train_data: pd.DataFrame, 
                target_column: str,
                test_data: Optional[pd.DataFrame] = None,
                config_file: Optional[str] = None) -> CompetitionFramework:
    """Quick training function for immediate use"""
    
    # Initialize framework
    framework = CompetitionFramework(config_file)
    
    # Prepare data
    X_train, y_train, X_test = framework.prepare_data(train_data, target_column, test_data)
    
    # Train models
    framework.train_models(X_train, y_train)
    
    # Create ensembles
    framework.create_ensembles(X_train, y_train)
    
    return framework

def create_default_config(filepath: str = 'competition_config.yaml'):
    """Create a default configuration file"""
    config = CompetitionConfig()
    config.save_config(filepath)
    print(f"Default configuration saved to {filepath}")

if __name__ == "__main__":
    # Example usage
    print("AI Competition Toolkit - Ready for use!")
    print("Use quick_train() for immediate training or CompetitionFramework() for full control.")
    print("Run create_default_config() to generate a configuration file.")