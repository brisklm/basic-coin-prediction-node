# AI Competition Toolkit 🚀

A comprehensive, autonomous framework for machine learning competitions that automatically analyzes competition requirements, learns from GitHub repositories, optimizes code using MCP (Model Context Protocol), and builds high-performance models without manual intervention.

## 🚀 Autonomous Features

### 🤖 **Intelligent Competition Analysis**
- **Web Scraping**: Automatically reads competition requirements, rules, and evaluation metrics from Kaggle, DrivenData, and other platforms
- **Problem Detection**: Identifies problem type, target column, and optimal evaluation strategy
- **Rule Compliance**: Ensures solutions follow competition-specific rules and constraints

### 📚 **GitHub Repository Learning**
- **Best Practice Extraction**: Analyzes top-performing solutions from GitHub repositories
- **Pattern Recognition**: Identifies common techniques, model architectures, and preprocessing strategies
- **Code Analysis**: Parses Python/R code to extract successful approaches and methodologies

### 🔮 **Cyclical MCP Optimization**
- **Dual MCP Servers**: Optimizer and Evaluator servers work in cyclical harmony
- **Iterative Improvement**: Continuous optimization until convergence criteria are met
- **Autonomous Code Generation**: Leverages Model Context Protocol for intelligent code optimization
- **Performance Evaluation**: AI-powered assessment of model quality and convergence
- **Customizable Thresholds**: Flexible stopping criteria and convergence detection
- **Competition-Specific Tuning**: Adapts strategies based on competition requirements and best practices

### 🎯 **Core ML Capabilities**
- **Automated Pipeline**: Complete end-to-end automation from raw data to competition submission
- **Smart Preprocessing**: Handles missing values, categorical encoding, feature scaling, and outlier detection
- **Advanced Feature Engineering**: Automatic generation of polynomial, interaction, and statistical features
- **Multi-Model Support**: LightGBM, XGBoost, CatBoost, Random Forest, Linear models, and SVM
- **Hyperparameter Optimization**: Bayesian optimization using Optuna for optimal model performance
- **Ensemble Methods**: Voting, stacking, and weighted averaging for superior predictions
- **Auto-Detection**: Automatically detects problem type (classification/regression) and appropriate metrics
- **Cross-Validation**: Robust evaluation with stratified/standard k-fold validation
- **Model Persistence**: Save and load trained models for reuse
- **Competition Ready**: Direct submission file generation

## 📦 Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- numpy, pandas, scikit-learn
- xgboost, lightgbm, catboost
- optuna, hyperopt, bayesian-optimization
- matplotlib, seaborn, plotly
- shap, feature-engine, imbalanced-learn
- joblib, tqdm, pyyaml

## 🏃‍♂️ Quick Start

### 🤖 Autonomous Mode with Cyclical MCP Optimization (Recommended)

```python
from enhanced_competition_toolkit import autonomous_competition_solution
from cyclical_mcp_system import CyclicalConfig

# Complete autonomous solution with cyclical MCP optimization!
solution_report = autonomous_competition_solution(
    competition_url="https://www.kaggle.com/competitions/your-competition",
    train_data_path="train.csv",
    test_data_path="test.csv",
    sample_submission_path="sample_submission.csv",  # optional
    github_token="your_github_token",  # optional, for enhanced analysis
    mcp_api_key="your_mcp_api_key",     # optional, for AI-powered optimization
    enable_cyclical_optimization=True,   # Enable cyclical MCP optimization
    cyclical_config=CyclicalConfig(
        max_iterations=10,
        convergence_threshold=0.001,
        absolute_performance_threshold=0.85  # Stop at 85% performance
    )
)

# The system will autonomously:
# 1. 🔍 Analyze the competition requirements automatically
# 2. 📚 Study successful GitHub repositories for best practices
# 3. 🔄 Run cyclical optimization with dual MCP servers:
#    • MCP Optimizer: Generates improvement strategies
#    • MCP Evaluator: Assesses performance and convergence
#    • Iterates until convergence criteria are met
# 4. 🚀 Generate optimized submission with best performance
```

### ⚡ Ultra-Quick Mode

```python
from enhanced_competition_toolkit import quick_competition_solution

# One-line solution
submission_file = quick_competition_solution(
    competition_url="https://www.kaggle.com/competitions/your-competition",
    train_csv="train.csv",
    test_csv="test.csv"
)
print(f"Submission ready: {submission_file}")
```

### 🛠️ Traditional Usage (Manual Control)

```python
from ai_competition_toolkit import quick_train
import pandas as pd

# Load your data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Train models automatically
framework = quick_train(train_data, target_column='target', test_data=test_data)

# Generate submission
submission_format = pd.read_csv('sample_submission.csv')
framework.generate_submission(test_data, submission_format, 'my_submission.csv')
```

### 🎯 Enhanced Autonomous Usage with Cyclical MCP

```python
from enhanced_competition_toolkit import EnhancedCompetitionFramework
from cyclical_mcp_system import CyclicalConfig

# Initialize with cyclical MCP optimization
framework = EnhancedCompetitionFramework(
    competition_url="https://www.kaggle.com/competitions/your-competition",
    github_token="your_token",           # optional
    mcp_api_key="your_key",             # optional
    auto_analyze=True,                  # enables autonomous analysis
    enable_cyclical_optimization=True,  # enables cyclical MCP
    cyclical_config=CyclicalConfig(
        max_iterations=8,
        convergence_threshold=0.002,
        consecutive_no_improvement=3,
        absolute_performance_threshold=0.9
    )
)

# The framework automatically:
# - Scrapes competition requirements and rules
# - Analyzes successful GitHub repositories
# - Runs cyclical optimization with dual MCP servers
# - Applies iterative improvements until convergence

# Auto-train with cyclical optimization
training_report = framework.auto_train_with_optimization(train_data, test_data=test_data)

# Check cyclical optimization results
cyclical_results = framework.get_cyclical_optimization_results()
if cyclical_results:
    print(f"🏆 Best performance: {cyclical_results['optimization_summary']['best_performance']}")
    print(f"🔄 Iterations: {cyclical_results['optimization_summary']['total_iterations']}")
    print(f"📈 Converged: {cyclical_results['optimization_summary']['convergence_achieved']}")

# Generate optimized submission
framework.generate_competition_submission(
    test_data, 
    sample_submission_path="sample_submission.csv",
    output_filename="cyclically_optimized_submission.csv"
)
```

### 🛠️ Advanced Manual Usage

```python
from ai_competition_toolkit import CompetitionFramework

# Initialize with custom configuration
framework = CompetitionFramework('my_config.yaml')

# Prepare data
X_train, y_train, X_test = framework.prepare_data(train_data, 'target', test_data)

# Train models with optimization
framework.train_models(X_train, y_train)

# Create ensemble models
framework.create_ensembles(X_train, y_train)

# Make predictions using best ensemble
predictions = framework.predict(X_test, use_ensemble='stacking')
```

## 🛠️ Configuration

Create a custom configuration file:

```python
from ai_competition_toolkit import create_default_config

create_default_config('my_config.yaml')
```

### Configuration Options

```yaml
problem_type: 'auto'  # 'classification', 'regression', 'auto'
target_column: 'target'
metric: 'auto'  # Will be auto-detected
cv_folds: 5
random_state: 42
max_trials: 100  # Hyperparameter optimization trials

# Feature engineering
feature_selection: true
feature_engineering: true

# Ensemble methods
ensemble_methods: ['voting', 'stacking']

# Model selection
models:
  lgb: true
  xgb: true
  catboost: true
  rf: true
  lr: true
  svm: false

# Preprocessing
preprocessing:
  handle_missing: true
  encode_categorical: true
  scale_features: true
  remove_outliers: false
```

## 🏆 Competition Workflow

### 1. Data Preparation
```python
framework = CompetitionFramework()

# The framework automatically:
# - Detects problem type (classification/regression)
# - Handles missing values
# - Encodes categorical variables
# - Scales numerical features
# - Engineers new features
# - Selects most informative features

X_train, y_train, X_test = framework.prepare_data(train_df, 'target', test_df)
```

### 2. Model Training & Optimization
```python
# Trains multiple models with hyperparameter optimization
framework.train_models(X_train, y_train)

# Each model is optimized using Bayesian optimization:
# - LightGBM: n_estimators, learning_rate, max_depth, etc.
# - XGBoost: Similar parameters optimized
# - CatBoost: Iterations, learning_rate, depth, etc.
# - Random Forest: n_estimators, max_depth, min_samples_split, etc.
# - Linear Models: Regularization parameters
```

### 3. Ensemble Creation
```python
# Creates powerful ensemble models
framework.create_ensembles(X_train, y_train)

# Available ensemble methods:
# - Voting: Simple/weighted average of predictions
# - Stacking: Meta-model learns optimal combination
# - Custom: Weighted averaging with performance-based weights
```

### 4. Prediction & Submission
```python
# Generate final predictions
predictions = framework.predict(X_test, use_ensemble='stacking')

# Create submission file
framework.generate_submission(X_test, submission_format, 'submission.csv')
```

## 📊 Model Performance

The toolkit automatically evaluates models using appropriate metrics:

- **Classification**: ROC-AUC, Accuracy, F1-Score
- **Regression**: MSE, MAE, R²

Cross-validation ensures robust performance estimates.

## 🔧 Advanced Features

### Feature Engineering

```python
# Automatic feature generation:
# - Polynomial features (x², x³)
# - Interaction features (x₁ × x₂)
# - Statistical features (mean, std, min, max)
# - Domain-specific transformations

framework.feature_engineer.engineer_features(X, y)
```

### Hyperparameter Optimization

```python
# Bayesian optimization with Optuna
framework.model_optimizer.optimize_model('lgb', X, y, 'classification', 'roc_auc')

# Automatically explores:
# - Learning rates
# - Tree parameters
# - Regularization
# - Sampling parameters
```

### Model Persistence

```python
# Save entire framework
framework.save_model('my_model')

# Load for inference
framework.load_model('my_model_complete.pkl')
```

## 📈 Performance Tips

### 🤖 Autonomous Optimization
1. **Use Competition URLs**: Provide competition URLs for automatic requirement analysis
2. **GitHub Token**: Add GitHub token for enhanced repository analysis
3. **MCP Integration**: Use MCP API keys for AI-powered cyclical optimization
4. **Enable Cyclical Mode**: Use cyclical optimization for iterative improvement
5. **Configure Convergence**: Set custom thresholds and stopping criteria
6. **Let AI Decide**: Trust the autonomous cyclical process for optimal results

### 🎯 Manual Optimization
1. **Feature Engineering**: Enable for complex datasets
2. **Ensemble Methods**: Use stacking for best performance
3. **Hyperparameter Trials**: Increase for better optimization
4. **Cross-Validation**: More folds for stable estimates
5. **Model Selection**: Enable all models for diverse ensemble

### Performance Tuning

```python
# For maximum performance
framework.config.set('max_trials', 200)
framework.config.set('cv_folds', 10)
framework.config.set('feature_engineering', True)
framework.config.set('ensemble_methods', ['voting', 'stacking'])

# For faster iterations
framework.config.set('max_trials', 50)
framework.config.set('cv_folds', 3)
framework.config.set('models', {'lgb': True, 'xgb': True, 'catboost': False})
```

## 🎯 Use Cases

### 🏆 Competition Scenarios
- **Kaggle Competitions**: Full automation from URL to winning submission
- **DrivenData Challenges**: Automated analysis and optimization
- **Corporate ML Contests**: Rapid deployment with best practices
- **Academic Competitions**: Research-backed approach optimization

### 💼 Business Applications
- **Sales Forecasting**: Automated time-series and regression modeling
- **Customer Churn**: Classification with business rule integration
- **Demand Prediction**: Multi-model ensemble for inventory optimization
- **Risk Assessment**: Compliance-aware model development

### 🔬 Research & Development
- **Baseline Establishment**: Rapid prototype development with SOTA techniques
- **Method Comparison**: Systematic evaluation across multiple approaches
- **Best Practice Learning**: Automated extraction from successful projects
- **Educational**: Learning ML pipeline automation and optimization

## 📋 Examples

See `example_usage.py` for comprehensive examples:

1. **Binary Classification**: Customer churn prediction
2. **Regression**: Sales forecasting
3. **Custom Configuration**: Tailored settings
4. **Competition Simulation**: End-to-end workflow
5. **Model Comparison**: Performance analysis

## 🤝 Contributing

Contributions are welcome! Priority areas:

### 🤖 Autonomous Intelligence
- Enhanced competition platform support (Codalab, AIcrowd, etc.)
- Improved GitHub repository analysis and pattern recognition
- Advanced MCP integration and optimization strategies
- Multi-language code analysis (R, Julia, etc.)

### 🚀 Core ML Capabilities
- Neural network integration (PyTorch, TensorFlow)
- Time series specific features and models
- Computer vision and NLP preprocessing pipelines
- GPU acceleration and distributed training

### 🔧 Platform & Tools
- Cloud deployment automation (AWS, GCP, Azure)
- Real-time monitoring and A/B testing
- Advanced visualization and interpretability
- Integration with MLOps platforms

## 📄 License

MIT License - feel free to use in competitions and commercial projects.

## 🎉 Autonomous Intelligence

This enhanced toolkit represents a breakthrough in automated machine learning competition solving:

### 🧠 **Intelligent Automation**
- **Zero-Configuration**: Automatically configures based on competition analysis
- **Best Practice Learning**: Continuously learns from top-performing solutions
- **Adaptive Optimization**: Uses AI to optimize code and strategies
- **Rule Compliance**: Ensures adherence to competition-specific requirements

### 🏆 **Competition-Ready Features**
- **Multi-Platform Support**: Works with Kaggle, DrivenData, and custom competitions
- **Submission Optimization**: Generates optimized predictions and submission files
- **Performance Tracking**: Comprehensive logging and performance analysis
- **Reproducible Results**: Deterministic pipelines with full auditability

### 🚀 **Next-Generation ML**
Built on winning strategies from top competitors and enhanced with:
- **Autonomous requirement analysis**
- **AI-powered code optimization**
- **Community knowledge integration**
- **Competition-specific fine-tuning**

**Start competing at the grandmaster level with just a competition URL!**

---

## 🎯 Quick Command Line Usage

```bash
# Full autonomous solution with cyclical MCP optimization
python enhanced_competition_toolkit.py \
  "https://www.kaggle.com/competitions/your-competition" \
  train.csv \
  test.csv \
  sample_submission.csv

# For ultimate performance with cyclical optimization:
python ultimate_competition_example.py

# The system handles everything else automatically with dual MCP servers! 🚀
```

## 🔄 Cyclical MCP Architecture

The toolkit features a revolutionary **dual MCP server cyclical optimization system**:

### 🏗️ **Architecture Components**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Competition   │    │  GitHub Repo    │    │   Cyclical MCP  │
│    Analysis     │────│    Analysis     │────│   Orchestrator  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                              ┌────────┴────────┐
                                              │                 │
                                    ┌─────────▼──────┐ ┌───────▼─────────┐
                                    │ MCP Optimizer  │ │ MCP Evaluator   │
                                    │    Server      │ │    Server       │
                                    │                │ │                 │
                                    │ • Strategy Gen │ │ • Performance   │
                                    │ • Code Optim   │ │ • Convergence   │
                                    │ • Improvement  │ │ • Assessment    │
                                    └─────────┬──────┘ └───────┬─────────┘
                                              │                │
                                              └────────┬───────┘
                                                       │
                                              ┌────────▼────────┐
                                              │  Competition    │
                                              │   Framework     │
                                              │  (Optimized)    │
                                              └─────────────────┘
```

### 🔄 **Cyclical Optimization Process**

1. **Initial Training**: Base model training with autonomous configuration
2. **MCP Evaluation**: Current solution assessment and bottleneck identification  
3. **MCP Optimization**: Strategy generation and improvement recommendations
4. **Implementation**: Apply optimizations to framework configuration
5. **Performance Check**: Evaluate improvements and convergence criteria
6. **Iteration**: Repeat until convergence or maximum iterations reached

### ⚙️ **Convergence Criteria (Customizable)**

```python
cyclical_config = CyclicalConfig(
    max_iterations=10,                          # Maximum optimization cycles
    convergence_threshold=0.001,                # Minimum improvement threshold
    consecutive_no_improvement=3,               # Stop after N iterations without improvement
    relative_improvement_threshold=0.005,       # Relative improvement requirement
    absolute_performance_threshold=0.85,        # Stop when target performance reached
    performance_metric="cv_score",              # Primary optimization metric
    timeout_per_iteration=600                   # Maximum time per iteration (seconds)
)
```