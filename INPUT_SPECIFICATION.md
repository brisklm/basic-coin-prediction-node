# 🎯 AI Competition Toolkit - Complete Input Specification

## 📋 **Input Types Overview**

The AI Competition Toolkit supports multiple input types for maximum flexibility and automation. Here's a comprehensive guide to all supported inputs:

## 🏆 **1. Competition Inputs**

### **Competition URL (Primary)**
```python
competition_url = "https://www.kaggle.com/competitions/titanic"
competition_url = "https://www.drivendata.org/competitions/44/"
competition_url = "https://codalab.lisn.upsaclay.fr/competitions/123"
```

**Supported Platforms:**
- ✅ **Kaggle**: `kaggle.com/competitions/*`
- ✅ **DrivenData**: `drivendata.org/competitions/*`
- 🔄 **Codalab**: `codalab.lisn.upsaclay.fr/competitions/*` (planned)
- 🔄 **AIcrowd**: `aicrowd.com/challenges/*` (planned)
- 🔄 **Custom**: Any competition with structured data

**What the system extracts:**
- Competition title and description
- Problem type (classification/regression)
- Evaluation metric
- Submission format
- Rules and constraints
- Deadline information
- Data description

## 📊 **2. Data Inputs**

### **Training Data**
```python
# CSV file path
train_data_path = "train.csv"
train_data_path = "/path/to/competition/data/train.csv"
train_data_path = "https://url-to-data/train.csv"  # URL support

# Pandas DataFrame (direct)
train_data = pd.read_csv("train.csv")

# Multiple file formats
train_data_path = "train.parquet"  # Parquet
train_data_path = "train.json"     # JSON
train_data_path = "train.xlsx"     # Excel
```

### **Test Data**
```python
test_data_path = "test.csv"
test_data_path = "/path/to/test.csv"
test_data = pd.read_csv("test.csv")  # Direct DataFrame
```

### **Sample Submission**
```python
sample_submission_path = "sample_submission.csv"
sample_submission_path = None  # Auto-generate format
```

### **Validation Data (Optional)**
```python
validation_data_path = "validation.csv"  # Optional holdout set
```

## 🐙 **3. GitHub Repository Inputs**

### **Reference Repository URLs**
```python
# Single repository
github_repos = ["https://github.com/user/winning-solution"]

# Multiple repositories
github_repos = [
    "https://github.com/user1/solution-1st-place",
    "https://github.com/user2/solution-2nd-place", 
    "https://github.com/user3/eda-analysis"
]

# Automatic search (recommended)
github_search_query = "kaggle titanic machine learning"
github_search_query = "house prices regression competition"
```

**Repository Analysis Features:**
- ✅ **Code Pattern Extraction**: Successful ML patterns
- ✅ **Model Architecture Discovery**: Popular model choices
- ✅ **Feature Engineering Techniques**: Creative feature creation
- ✅ **Preprocessing Strategies**: Data cleaning approaches
- ✅ **Ensemble Methods**: Combination techniques
- ✅ **Performance Benchmarks**: Historical results

### **GitHub Authentication**
```python
github_token = "ghp_xxxxxxxxxxxxxxxxxxxx"  # For enhanced API access
github_token = None  # Public access only (limited)
```

## 🔧 **4. Configuration Inputs**

### **Competition Configuration**
```python
# YAML configuration file
config_file = "competition_config.yaml"
config_file = "/path/to/custom_config.yaml"

# Dictionary configuration
custom_config = {
    'problem_type': 'classification',
    'target_column': 'survived',
    'metric': 'roc_auc',
    'models': {'lgb': True, 'xgb': True, 'rf': True}
}
```

### **Cyclical MCP Configuration**
```python
from cyclical_mcp_system import CyclicalConfig

cyclical_config = CyclicalConfig(
    max_iterations=10,
    convergence_threshold=0.001,
    consecutive_no_improvement=3,
    absolute_performance_threshold=0.85,
    performance_metric="cv_score",
    timeout_per_iteration=600,
    optimizer_config={
        "model": "claude-3-sonnet",
        "temperature": 0.7,
        "max_tokens": 6000
    },
    evaluator_config={
        "model": "claude-3-sonnet", 
        "temperature": 0.2,
        "max_tokens": 4000
    }
)
```

## 🤖 **5. AI Service Inputs**

### **MCP API Keys**
```python
# Anthropic Claude
anthropic_api_key = "sk-ant-xxxxxxxxxxxxxxxxxxxxx"

# OpenAI GPT
openai_api_key = "sk-xxxxxxxxxxxxxxxxxxxxx"

# Combined MCP key (either service)
mcp_api_key = anthropic_api_key  # or openai_api_key
```

### **AI Model Selection**
```python
optimizer_model = "claude-3-sonnet"      # claude-3-sonnet, gpt-4, gpt-4-turbo
evaluator_model = "claude-3-haiku"       # claude-3-haiku, gpt-3.5-turbo
temperature = 0.7                        # Creativity level (0.0-1.0)
max_tokens = 4000                        # Response length limit
```

## 📂 **6. Project Structure Inputs**

### **Project Root Directory**
```python
project_root = "."                       # Current directory
project_root = "/path/to/project"        # Absolute path
project_root = "../competition-project"   # Relative path
```

### **Output Directory**
```python
output_dir = "./results"                 # Default output location
output_dir = "/path/to/results"          # Custom output path
output_dir = f"./results_{competition_name}"  # Dynamic naming
```

## 🎯 **7. Target and Feature Specification**

### **Target Column**
```python
target_column = "target"                 # Default
target_column = "survived"               # Titanic example
target_column = "SalePrice"              # House prices example
target_column = None                     # Auto-detect from competition
```

### **Feature Selection**
```python
# Include specific features
include_features = ["age", "fare", "pclass"]

# Exclude specific features  
exclude_features = ["name", "ticket", "cabin"]

# Feature types
feature_types = {
    "numerical": ["age", "fare"],
    "categorical": ["sex", "embarked"],
    "datetime": ["date"]
}
```

## 🔄 **8. Processing Options**

### **Preprocessing Options**
```python
preprocessing_config = {
    'handle_missing': True,
    'encode_categorical': True,
    'scale_features': True,
    'remove_outliers': False,
    'feature_engineering': True,
    'feature_selection': True
}
```

### **Model Training Options**
```python
training_config = {
    'cv_folds': 5,
    'max_trials': 100,
    'random_state': 42,
    'models': {
        'lgb': True,
        'xgb': True,
        'catboost': True,
        'rf': True,
        'lr': True,
        'svm': False
    }
}
```

### **Ensemble Options**
```python
ensemble_config = {
    'methods': ['voting', 'stacking'],
    'voting_weights': None,  # Auto-determine
    'stacking_meta_model': 'lr',
    'blend_method': 'geometric_mean'
}
```

## 🚀 **Usage Examples by Input Type**

### **Minimal Input (URL Only)**
```python
from enhanced_competition_toolkit import quick_competition_solution

# Just competition URL - everything else is automated
submission = quick_competition_solution(
    competition_url="https://www.kaggle.com/competitions/titanic",
    train_csv="train.csv",
    test_csv="test.csv"
)
```

### **Standard Input (URL + Data)**
```python
from enhanced_competition_toolkit import autonomous_competition_solution

solution = autonomous_competition_solution(
    competition_url="https://www.kaggle.com/competitions/house-prices",
    train_data_path="train.csv",
    test_data_path="test.csv",
    sample_submission_path="sample_submission.csv"
)
```

### **Enhanced Input (URL + GitHub + MCP)**
```python
solution = autonomous_competition_solution(
    competition_url="https://www.kaggle.com/competitions/titanic",
    train_data_path="train.csv", 
    test_data_path="test.csv",
    github_token="ghp_xxxxxxxxxxxxxxxxxxxx",
    mcp_api_key="sk-ant-xxxxxxxxxxxxxxxxxxxxx",
    enable_cyclical_optimization=True
)
```

### **Advanced Input (Full Configuration)**
```python
from cyclical_mcp_system import CyclicalConfig

solution = autonomous_competition_solution(
    # Competition inputs
    competition_url="https://www.kaggle.com/competitions/advanced-regression",
    train_data_path="train.csv",
    test_data_path="test.csv", 
    sample_submission_path="sample_submission.csv",
    
    # GitHub inputs
    github_token="ghp_xxxxxxxxxxxxxxxxxxxx",
    github_repos=[
        "https://github.com/user1/1st-place-solution",
        "https://github.com/user2/feature-engineering-guide"
    ],
    
    # AI service inputs
    mcp_api_key="sk-ant-xxxxxxxxxxxxxxxxxxxxx",
    
    # Configuration inputs
    enable_cyclical_optimization=True,
    cyclical_config=CyclicalConfig(
        max_iterations=15,
        convergence_threshold=0.0005,
        absolute_performance_threshold=0.90,
        optimizer_config={"model": "claude-3-opus"},
        evaluator_config={"model": "claude-3-sonnet"}
    ),
    
    # Output inputs
    output_dir="./advanced_competition_results"
)
```

### **Framework Integration Input**
```python
from enhanced_competition_toolkit import EnhancedCompetitionFramework

framework = EnhancedCompetitionFramework(
    competition_url="https://www.kaggle.com/competitions/nlp-challenge",
    github_token="ghp_xxxxxxxxxxxxxxxxxxxx",
    mcp_api_key="sk-xxxxxxxxxxxxxxxxxxxxx",
    config_file="nlp_config.yaml",
    enable_cyclical_optimization=True,
    cyclical_config=cyclical_config
)

# Train with custom data
X_train, y_train, X_test = framework.prepare_data(
    train_data=custom_train_df,
    target_column="sentiment",
    test_data=custom_test_df
)

training_report = framework.auto_train_with_optimization(
    train_data=custom_train_df,
    test_data=custom_test_df
)
```

### **Direct Cyclical Optimization Input**
```python
from cyclical_mcp_system import run_cyclical_optimization

results = await run_cyclical_optimization(
    competition_url="https://www.kaggle.com/competitions/time-series",
    train_data_path="time_series_train.csv",
    validation_data_path="time_series_val.csv",
    config=CyclicalConfig(max_iterations=20),
    optimizer_api_key="sk-ant-xxxxxxxxxxxxxxxxxxxxx",
    evaluator_api_key="sk-xxxxxxxxxxxxxxxxxxxxx",  # Can use different keys
    output_dir="./time_series_optimization"
)
```

## 📋 **Input Validation & Error Handling**

### **Automatic Input Validation**
```python
# The system automatically validates:
✅ Competition URL accessibility
✅ Data file existence and format
✅ GitHub repository accessibility
✅ API key validity
✅ Configuration parameter correctness
✅ Feature/target column existence
✅ Data type compatibility
```

### **Fallback Mechanisms**
```python
# When inputs are missing or invalid:
🔄 Competition URL → Fallback to manual configuration
🔄 GitHub repos → Skip repository analysis
🔄 MCP API keys → Use rule-based optimization
🔄 Sample submission → Auto-generate format
🔄 Config file → Use intelligent defaults
```

## 🎯 **Input Prioritization**

### **Essential Inputs (Required)**
1. **Training Data**: CSV file or DataFrame
2. **Test Data**: CSV file or DataFrame
3. **Target Column**: Name or auto-detection

### **Highly Recommended Inputs**
1. **Competition URL**: Enables autonomous analysis
2. **Sample Submission**: Ensures correct format

### **Performance Enhancing Inputs**
1. **MCP API Key**: Enables AI-powered optimization
2. **GitHub Token**: Enhanced repository analysis
3. **Cyclical Config**: Custom optimization parameters

### **Optional Inputs**
1. **Custom Config**: Override defaults
2. **GitHub Repos**: Specific reference solutions
3. **Validation Data**: Additional evaluation
4. **Output Directory**: Custom result location

## 🚀 **Quick Start Input Templates**

### **Beginner Template**
```python
# Minimal input for getting started
solution = autonomous_competition_solution(
    competition_url="COMPETITION_URL_HERE",
    train_data_path="train.csv",
    test_data_path="test.csv"
)
```

### **Intermediate Template**
```python
# Enhanced input with cyclical optimization
solution = autonomous_competition_solution(
    competition_url="COMPETITION_URL_HERE",
    train_data_path="train.csv",
    test_data_path="test.csv",
    sample_submission_path="sample_submission.csv",
    mcp_api_key="YOUR_API_KEY_HERE",
    enable_cyclical_optimization=True
)
```

### **Advanced Template**
```python
# Full-featured input with all options
solution = autonomous_competition_solution(
    competition_url="COMPETITION_URL_HERE",
    train_data_path="train.csv",
    test_data_path="test.csv",
    sample_submission_path="sample_submission.csv",
    github_token="YOUR_GITHUB_TOKEN",
    mcp_api_key="YOUR_MCP_API_KEY", 
    enable_cyclical_optimization=True,
    cyclical_config=CyclicalConfig(
        max_iterations=10,
        absolute_performance_threshold=0.85
    ),
    output_dir="./my_competition_results"
)
```

This comprehensive input system makes the AI Competition Toolkit extremely flexible and powerful, supporting everything from simple beginner use cases to advanced professional deployment scenarios.