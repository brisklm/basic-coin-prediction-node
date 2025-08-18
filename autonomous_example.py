"""
Autonomous AI Competition Toolkit - Example Usage
================================================

Demonstrates the enhanced toolkit with automatic competition analysis,
GitHub repository learning, and MCP-powered optimization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Import the enhanced toolkit
from enhanced_competition_toolkit import (
    EnhancedCompetitionFramework,
    autonomous_competition_solution,
    quick_competition_solution
)

def demo_autonomous_analysis():
    """Demonstrate autonomous competition analysis"""
    print("=" * 60)
    print("DEMO 1: Autonomous Competition Analysis")
    print("=" * 60)
    
    # Example with Titanic competition (well-known Kaggle competition)
    competition_url = "https://www.kaggle.com/competitions/titanic"
    
    print(f"🔍 Analyzing competition: {competition_url}")
    
    # Initialize with automatic analysis
    framework = EnhancedCompetitionFramework(
        competition_url=competition_url,
        auto_analyze=True
    )
    
    # Get insights from analysis
    insights = framework.get_competition_insights()
    
    print("\n📊 Competition Insights:")
    print(f"  Title: {insights['competition_info']['title']}")
    print(f"  Problem Type: {insights['competition_info']['problem_type']}")
    print(f"  Evaluation Metric: {insights['competition_info']['evaluation_metric']}")
    print(f"  Target Column: {insights['competition_info']['target_column']}")
    print(f"  Repositories Analyzed: {insights['repositories_analyzed']}")
    
    print("\n💡 Key Recommendations:")
    for i, rec in enumerate(insights['recommendations'][:3], 1):
        print(f"  {i}. {rec}")
    
    print("\n🏆 Best Practices Found:")
    for practice in insights['best_practices']:
        print(f"  • {practice}")
    
    return framework

def demo_github_learning():
    """Demonstrate GitHub repository learning"""
    print("\n" + "=" * 60)
    print("DEMO 2: GitHub Repository Learning")
    print("=" * 60)
    
    from competition_analyzer import GitHubAnalyzer
    
    # Analyze repositories for a specific competition type
    analyzer = GitHubAnalyzer()
    
    print("🔍 Searching for classification competition solutions...")
    
    # Search for classification competitions on GitHub
    search_query = "machine learning classification competition kaggle"
    
    try:
        analysis = analyzer.analyze_competition_repos(search_query, max_repos=3)
        
        print(f"\n📚 Analyzed {len(analysis['repositories'])} repositories")
        
        # Show common patterns
        common_patterns = analysis.get('common_patterns', {})
        
        if 'most_common_models' in common_patterns:
            print("\n🤖 Most Popular Models:")
            for model, count in common_patterns['most_common_models'][:5]:
                print(f"  • {model}: {count} repositories")
        
        if 'most_common_techniques' in common_patterns:
            print("\n🛠️ Most Common Techniques:")
            for technique, count in common_patterns['most_common_techniques'][:5]:
                print(f"  • {technique}: {count} repositories")
        
        # Show repository details
        print("\n📖 Repository Analysis:")
        for repo in analysis['repositories'][:2]:
            print(f"  📁 {repo['name']} (⭐ {repo['stars']})")
            print(f"     Language: {repo['language']}")
            print(f"     Models: {', '.join(repo['models_used'][:3]) if repo['models_used'] else 'None detected'}")
            print(f"     Techniques: {', '.join(repo['techniques'][:3]) if repo['techniques'] else 'None detected'}")
            print()
    
    except Exception as e:
        print(f"⚠️ GitHub analysis failed (this is normal without API token): {e}")
        print("💡 Tip: Provide a GitHub token for full repository analysis")

def demo_mcp_optimization():
    """Demonstrate MCP-powered code optimization"""
    print("\n" + "=" * 60)
    print("DEMO 3: MCP Code Optimization")
    print("=" * 60)
    
    from competition_analyzer import MCPOptimizer, CompetitionInfo
    
    # Create a sample competition info
    competition_info = CompetitionInfo(
        title="Binary Classification Challenge",
        description="Predict binary outcomes based on features",
        problem_type="classification",
        evaluation_metric="roc_auc",
        submission_format="csv",
        deadline="2024-12-31",
        rules=["Use only provided data", "Maximum 5 submissions per day"],
        data_description="Features include numerical and categorical variables",
        target_column="target",
        sample_submission_format={}
    )
    
    # Sample code to optimize
    original_code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('train.csv')
X = df.drop('target', axis=1)
y = df['target']

# Simple split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
"""
    
    print("🔧 Original Code:")
    print(original_code)
    
    # Initialize MCP optimizer
    mcp_optimizer = MCPOptimizer()
    
    # Mock GitHub analysis results
    github_analysis = {
        'common_patterns': {
            'most_common_models': [('LGBMClassifier', 15), ('XGBClassifier', 12), ('RandomForestClassifier', 8)],
            'most_common_techniques': [('cross_val_score', 20), ('StratifiedKFold', 18), ('GridSearchCV', 15)]
        }
    }
    
    print("\n🤖 Applying MCP optimization...")
    
    # Optimize code (this will use rule-based optimization as fallback)
    optimized_code = mcp_optimizer.optimize_competition_code(
        competition_info, github_analysis, original_code
    )
    
    print("\n✨ Optimized Code:")
    print(optimized_code)

def demo_complete_autonomous_solution():
    """Demonstrate complete autonomous solution pipeline"""
    print("\n" + "=" * 60)
    print("DEMO 4: Complete Autonomous Solution")
    print("=" * 60)
    
    # Create sample data for demonstration
    print("📊 Creating sample competition data...")
    
    # Generate synthetic competition data
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(0, 1, n_samples),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature_4': np.random.uniform(0, 100, n_samples),
        'feature_5': np.random.exponential(2, n_samples)
    }
    
    # Create target (classification)
    target = (data['feature_1'] + data['feature_2'] + 
             np.where(data['feature_3'] == 'A', 1, 0) > 0).astype(int)
    
    # Create DataFrames
    train_df = pd.DataFrame(data)
    train_df['target'] = target
    
    # Test data (without target)
    test_data = {key: np.random.normal(0, 1, 200) if key in ['feature_1', 'feature_2'] 
                 else np.random.choice(['A', 'B', 'C'], 200) if key == 'feature_3'
                 else np.random.uniform(0, 100, 200) if key == 'feature_4'
                 else np.random.exponential(2, 200)
                 for key in data.keys()}
    
    test_df = pd.DataFrame(test_data)
    
    # Save to CSV files
    train_df.to_csv('demo_train.csv', index=False)
    test_df.to_csv('demo_test.csv', index=False)
    
    # Create sample submission format
    sample_submission = pd.DataFrame({
        'id': range(len(test_df)),
        'target': 0
    })
    sample_submission.to_csv('demo_sample_submission.csv', index=False)
    
    print(f"✅ Created demo data: {len(train_df)} training samples, {len(test_df)} test samples")
    
    # Initialize enhanced framework
    print("\n🤖 Initializing autonomous framework...")
    
    framework = EnhancedCompetitionFramework(
        # competition_url="https://www.kaggle.com/competitions/titanic",  # Comment out for demo
        auto_analyze=False  # Skip web scraping for demo
    )
    
    # Manually configure for demo
    framework.config.set('problem_type', 'classification')
    framework.config.set('target_column', 'target')
    framework.config.set('metric', 'roc_auc')
    framework.config.set('max_trials', 10)  # Quick demo
    framework.config.set('cv_folds', 3)
    
    print("🚀 Starting autonomous training...")
    
    # Auto-train with optimizations
    training_report = framework.auto_train_with_optimization(
        train_data=train_df,
        target_column='target',
        test_data=test_df
    )
    
    print("\n📋 Training Report:")
    for key, value in training_report.items():
        print(f"  {key}: {value}")
    
    # Generate submission
    print("\n🎯 Generating competition submission...")
    
    submission_file = framework.generate_competition_submission(
        test_data=test_df,
        sample_submission_path='demo_sample_submission.csv',
        output_filename='demo_submission.csv'
    )
    
    print(f"✅ Submission generated: {submission_file}")
    
    # Show sample predictions
    submission_df = pd.read_csv(submission_file)
    print(f"\n📊 Sample predictions:")
    print(submission_df.head(10))
    
    # Clean up demo files
    demo_files = ['demo_train.csv', 'demo_test.csv', 'demo_sample_submission.csv']
    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)
    
    print("\n🧹 Cleaned up demo files")

def demo_quick_solution():
    """Demonstrate ultra-quick solution mode"""
    print("\n" + "=" * 60)
    print("DEMO 5: Ultra-Quick Solution Mode")
    print("=" * 60)
    
    # Create minimal test data
    np.random.seed(42)
    
    # Quick training data
    quick_train = pd.DataFrame({
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    # Quick test data
    quick_test = pd.DataFrame({
        'x1': np.random.randn(50),
        'x2': np.random.randn(50)
    })
    
    # Save files
    quick_train.to_csv('quick_train.csv', index=False)
    quick_test.to_csv('quick_test.csv', index=False)
    
    print("⚡ Using ultra-quick solution mode...")
    
    try:
        # This would normally use a real competition URL
        submission_file = quick_competition_solution(
            competition_url="https://example.com/competition",  # Demo URL
            train_csv='quick_train.csv',
            test_csv='quick_test.csv'
        )
        
        print(f"⚡ Quick solution complete! File: {submission_file}")
        
        # Clean up
        for file in ['quick_train.csv', 'quick_test.csv', submission_file]:
            if os.path.exists(file):
                os.remove(file)
    
    except Exception as e:
        print(f"⚠️ Quick solution demo failed (expected without real competition): {e}")
        
        # Clean up anyway
        for file in ['quick_train.csv', 'quick_test.csv']:
            if os.path.exists(file):
                os.remove(file)

def run_all_demos():
    """Run all demonstration examples"""
    print("🤖 Enhanced AI Competition Toolkit - Autonomous Capabilities Demo")
    print("=" * 70)
    
    try:
        # Demo 1: Autonomous analysis
        framework = demo_autonomous_analysis()
        
        # Demo 2: GitHub learning
        demo_github_learning()
        
        # Demo 3: MCP optimization
        demo_mcp_optimization()
        
        # Demo 4: Complete solution
        demo_complete_autonomous_solution()
        
        # Demo 5: Quick mode
        demo_quick_solution()
        
        print("\n" + "=" * 70)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print()
        print("🚀 Ready to use with real competitions!")
        print("💡 Try: python enhanced_competition_toolkit.py <competition_url> train.csv test.csv")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_demos()