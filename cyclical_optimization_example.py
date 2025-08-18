"""
Cyclical MCP Optimization Example
================================

Demonstrates the dual MCP server cyclical optimization system with
customizable convergence criteria and thresholds.
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

from cyclical_mcp_system import (
    CyclicalMCPOrchestrator,
    CyclicalConfig,
    run_cyclical_optimization
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_basic_cyclical_optimization():
    """Basic demonstration of cyclical optimization"""
    print("🔄 DEMO 1: Basic Cyclical Optimization")
    print("=" * 50)
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 800
    
    # Generate features with some complexity
    data = {
        'numeric_1': np.random.normal(0, 1, n_samples),
        'numeric_2': np.random.normal(2, 1.5, n_samples),
        'numeric_3': np.random.exponential(1, n_samples),
        'categorical_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'categorical_2': np.random.choice(['X', 'Y'], n_samples),
    }
    
    # Create target with some signal
    target = (
        0.5 * data['numeric_1'] +
        0.3 * data['numeric_2'] +
        0.2 * (data['categorical_1'] == 'A').astype(int) +
        np.random.normal(0, 0.1, n_samples)
    )
    target = (target > np.median(target)).astype(int)
    
    data['target'] = target
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    train_path = "demo_cyclical_train.csv"
    df.to_csv(train_path, index=False)
    
    print(f"📊 Created dataset: {df.shape[0]} samples, {df.shape[1]-1} features")
    
    # Configure cyclical optimization
    config = CyclicalConfig(
        max_iterations=5,
        convergence_threshold=0.005,
        min_improvement_threshold=0.01,
        consecutive_no_improvement=3,
        performance_metric="cv_score",
        save_intermediate_results=True
    )
    
    print("🔧 Configuration:")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Convergence threshold: {config.convergence_threshold}")
    print(f"  Min improvement: {config.min_improvement_threshold}")
    
    try:
        # Run cyclical optimization
        results = await run_cyclical_optimization(
            competition_url="https://demo.com/basic-classification",
            train_data_path=train_path,
            config=config,
            output_dir="./demo_basic_cyclical_results"
        )
        
        # Display results
        summary = results['optimization_summary']
        print(f"\n✅ Optimization completed!")
        print(f"🏆 Best performance: {summary['best_performance']:.4f}")
        print(f"🔄 Total iterations: {summary['total_iterations']}")
        print(f"⏱️ Total time: {summary['total_time_seconds']:.1f}s")
        print(f"📈 Convergence: {'Yes' if summary['convergence_achieved'] else 'No'}")
        
        # Show performance progression
        progression = results['performance_progression']
        print(f"\n📊 Performance progression:")
        for i, score in enumerate(progression, 1):
            print(f"  Iteration {i}: {score:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Basic demo failed: {e}")
        return None
    
    finally:
        # Cleanup
        import os
        if os.path.exists(train_path):
            os.remove(train_path)

async def demo_advanced_cyclical_optimization():
    """Advanced demonstration with custom thresholds"""
    print("\n🔄 DEMO 2: Advanced Cyclical Optimization")
    print("=" * 50)
    
    # Create more complex dataset
    np.random.seed(123)
    n_samples = 1200
    
    # Generate correlated features
    cov_matrix = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    features = np.random.multivariate_normal([0, 0, 0], cov_matrix, n_samples)
    
    data = {
        'feature_a': features[:, 0],
        'feature_b': features[:, 1], 
        'feature_c': features[:, 2],
        'feature_d': np.random.gamma(2, 2, n_samples),
        'feature_e': np.random.beta(2, 5, n_samples),
        'category_1': np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.3, 0.4, 0.3]),
        'category_2': np.random.choice(['red', 'blue', 'green'], n_samples),
    }
    
    # Complex target generation
    target = (
        0.4 * data['feature_a'] +
        0.3 * data['feature_b'] +
        0.2 * data['feature_c'] +
        0.1 * (data['category_1'] == 'high').astype(int) +
        np.random.normal(0, 0.15, n_samples)
    )
    target = (target > np.median(target)).astype(int)
    
    data['target'] = target
    
    # Create train/validation split
    df = pd.DataFrame(data)
    train_df = df.iloc[:800].copy()
    val_df = df.iloc[800:].copy()
    
    train_path = "demo_advanced_train.csv"
    val_path = "demo_advanced_val.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"📊 Training set: {train_df.shape[0]} samples")
    print(f"📊 Validation set: {val_df.shape[0]} samples")
    
    # Advanced configuration with strict convergence criteria
    config = CyclicalConfig(
        max_iterations=8,
        convergence_threshold=0.002,  # Stricter threshold
        min_improvement_threshold=0.005,
        consecutive_no_improvement=2,  # Stop faster
        relative_improvement_threshold=0.001,
        absolute_performance_threshold=0.85,  # Stop if we reach 85% performance
        performance_metric="cv_score",
        timeout_per_iteration=600,  # 10 minutes per iteration
        parallel_evaluation=True,
        save_intermediate_results=True,
        optimizer_config={
            "model": "claude-3-sonnet",
            "temperature": 0.8,  # More creative optimization
            "max_tokens": 5000
        },
        evaluator_config={
            "model": "claude-3-sonnet",
            "temperature": 0.2,  # More analytical evaluation
            "max_tokens": 3000
        }
    )
    
    print("🔧 Advanced Configuration:")
    print(f"  Convergence threshold: {config.convergence_threshold}")
    print(f"  Absolute performance target: {config.absolute_performance_threshold}")
    print(f"  Consecutive no-improvement limit: {config.consecutive_no_improvement}")
    print(f"  Parallel evaluation: {config.parallel_evaluation}")
    
    try:
        # Run advanced cyclical optimization
        results = await run_cyclical_optimization(
            competition_url="https://demo.com/advanced-classification",
            train_data_path=train_path,
            validation_data_path=val_path,
            config=config,
            output_dir="./demo_advanced_cyclical_results"
        )
        
        # Detailed results analysis
        summary = results['optimization_summary']
        print(f"\n✅ Advanced optimization completed!")
        print(f"🏆 Best performance: {summary['best_performance']:.4f}")
        print(f"🎯 Performance target: {config.absolute_performance_threshold}")
        print(f"🔄 Iterations completed: {summary['total_iterations']}/{config.max_iterations}")
        print(f"⏱️ Total time: {summary['total_time_seconds']:.1f}s")
        
        # Convergence analysis
        convergence = "Target reached" if summary['best_performance'] >= config.absolute_performance_threshold else \
                     "Converged" if summary['convergence_achieved'] else "Max iterations"
        print(f"🛑 Stop reason: {convergence}")
        
        # Performance and improvement progression
        perf_progression = results['performance_progression']
        improvement_progression = results['improvement_progression']
        
        print(f"\n📈 Detailed progression:")
        for i, (perf, improvement) in enumerate(zip(perf_progression, improvement_progression), 1):
            status = "🎯" if perf >= config.absolute_performance_threshold else \
                    "📈" if improvement > 0 else "📉"
            print(f"  {status} Iteration {i}: Performance={perf:.4f}, Improvement={improvement:+.4f}")
        
        # MCP server statistics
        mcp_stats = results['mcp_server_stats']
        print(f"\n🤖 MCP Server Statistics:")
        print(f"  Optimizer calls: {mcp_stats['optimizer_calls']}")
        print(f"  Evaluator calls: {mcp_stats['evaluator_calls']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Advanced demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Cleanup
        import os
        for file_path in [train_path, val_path]:
            if os.path.exists(file_path):
                os.remove(file_path)

async def demo_custom_convergence_criteria():
    """Demonstrate custom convergence criteria"""
    print("\n🔄 DEMO 3: Custom Convergence Criteria")
    print("=" * 50)
    
    # Quick dataset
    np.random.seed(999)
    n_samples = 600
    
    data = {
        'x1': np.random.randn(n_samples),
        'x2': np.random.randn(n_samples),
        'x3': np.random.choice(['P', 'Q', 'R'], n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    train_path = "demo_custom_train.csv"
    df.to_csv(train_path, index=False)
    
    # Test different convergence configurations
    convergence_configs = [
        {
            "name": "Quick Convergence",
            "config": CyclicalConfig(
                max_iterations=3,
                convergence_threshold=0.05,  # Very loose
                consecutive_no_improvement=1,
                performance_metric="cv_score"
            )
        },
        {
            "name": "Strict Convergence", 
            "config": CyclicalConfig(
                max_iterations=6,
                convergence_threshold=0.001,  # Very strict
                min_improvement_threshold=0.005,
                consecutive_no_improvement=2,
                relative_improvement_threshold=0.002,
                performance_metric="cv_score"
            )
        },
        {
            "name": "Target-Based Convergence",
            "config": CyclicalConfig(
                max_iterations=10,
                absolute_performance_threshold=0.75,  # Stop at 75% performance
                consecutive_no_improvement=5,
                performance_metric="cv_score"
            )
        }
    ]
    
    results_comparison = []
    
    for i, test_case in enumerate(convergence_configs):
        print(f"\n🧪 Testing {test_case['name']}:")
        config = test_case['config']
        
        print(f"  Max iterations: {config.max_iterations}")
        print(f"  Convergence threshold: {config.convergence_threshold}")
        print(f"  Absolute threshold: {config.absolute_performance_threshold}")
        print(f"  No improvement limit: {config.consecutive_no_improvement}")
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            results = await run_cyclical_optimization(
                competition_url=f"https://demo.com/custom-test-{i+1}",
                train_data_path=train_path,
                config=config,
                output_dir=f"./demo_custom_{i+1}_results"
            )
            
            end_time = asyncio.get_event_loop().time()
            
            summary = results['optimization_summary']
            
            test_result = {
                "name": test_case['name'],
                "iterations": summary['total_iterations'],
                "best_performance": summary['best_performance'],
                "time_seconds": end_time - start_time,
                "converged": summary['convergence_achieved']
            }
            
            results_comparison.append(test_result)
            
            print(f"    ✅ Completed: {test_result['iterations']} iterations, "
                  f"{test_result['best_performance']:.4f} performance, "
                  f"{test_result['time_seconds']:.1f}s")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            results_comparison.append({
                "name": test_case['name'],
                "iterations": 0,
                "best_performance": 0.0,
                "time_seconds": 0.0,
                "converged": False,
                "error": str(e)
            })
    
    # Comparison summary
    print(f"\n📊 CONVERGENCE CRITERIA COMPARISON:")
    print("-" * 70)
    print(f"{'Strategy':<25} {'Iterations':<12} {'Performance':<12} {'Time (s)':<10} {'Converged'}")
    print("-" * 70)
    
    for result in results_comparison:
        if "error" not in result:
            print(f"{result['name']:<25} {result['iterations']:<12} "
                  f"{result['best_performance']:<12.4f} {result['time_seconds']:<10.1f} "
                  f"{'Yes' if result['converged'] else 'No'}")
        else:
            print(f"{result['name']:<25} {'Failed':<12} {'N/A':<12} {'N/A':<10} {'N/A'}")
    
    print("-" * 70)
    
    # Cleanup
    import os
    if os.path.exists(train_path):
        os.remove(train_path)
    
    return results_comparison

async def demo_real_world_simulation():
    """Simulate real-world competition scenario"""
    print("\n🔄 DEMO 4: Real-World Competition Simulation")
    print("=" * 50)
    
    # Create realistic tabular dataset
    np.random.seed(2024)
    n_samples = 2000
    
    # Simulate customer data
    data = {
        'age': np.random.normal(35, 12, n_samples),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples),
        'debt_ratio': np.random.beta(2, 5, n_samples),
        'employment_length': np.random.exponential(5, n_samples),
        'education': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], 
                                    n_samples, p=[0.4, 0.35, 0.2, 0.05]),
        'home_ownership': np.random.choice(['rent', 'own', 'mortgage'], 
                                         n_samples, p=[0.3, 0.3, 0.4]),
        'state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'other'], 
                                n_samples, p=[0.15, 0.1, 0.1, 0.08, 0.57])
    }
    
    # Create realistic target (loan default prediction)
    risk_score = (
        -0.3 * (data['age'] - 35) / 12 +
        -0.4 * np.log(data['income'] / 50000) +
        -0.5 * (data['credit_score'] - 650) / 100 +
        0.6 * data['debt_ratio'] +
        -0.2 * np.log(data['employment_length'] + 1) +
        0.3 * (data['education'] == 'high_school').astype(int) +
        0.2 * (data['home_ownership'] == 'rent').astype(int) +
        np.random.normal(0, 0.3, n_samples)
    )
    
    # Convert to binary target (default probability)
    default_prob = 1 / (1 + np.exp(-risk_score))
    data['default'] = np.random.binomial(1, default_prob)
    
    # Create realistic train/validation/test splits
    df = pd.DataFrame(data)
    
    # Split by time (simulate time-based competition)
    train_df = df.iloc[:1200].copy()
    val_df = df.iloc[1200:1600].copy()
    test_df = df.iloc[1600:].copy()
    
    train_path = "real_world_train.csv"
    val_path = "real_world_val.csv"
    test_path = "real_world_test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"📊 Realistic loan default prediction dataset:")
    print(f"  Training: {train_df.shape[0]} samples")
    print(f"  Validation: {val_df.shape[0]} samples") 
    print(f"  Test: {test_df.shape[0]} samples")
    print(f"  Default rate: {data['default'].mean():.2%}")
    
    # Production-ready configuration
    config = CyclicalConfig(
        max_iterations=7,
        convergence_threshold=0.002,
        min_improvement_threshold=0.01,
        consecutive_no_improvement=3,
        relative_improvement_threshold=0.003,
        absolute_performance_threshold=0.82,  # 82% AUC target
        performance_metric="cv_score",
        timeout_per_iteration=900,  # 15 minutes per iteration
        parallel_evaluation=True,
        save_intermediate_results=True,
        optimizer_config={
            "model": "claude-3-sonnet",
            "temperature": 0.6,
            "max_tokens": 6000
        },
        evaluator_config={
            "model": "claude-3-sonnet", 
            "temperature": 0.1,  # Very analytical
            "max_tokens": 4000
        }
    )
    
    print(f"\n🔧 Production Configuration:")
    print(f"  Target AUC: {config.absolute_performance_threshold}")
    print(f"  Max optimization time: {config.max_iterations * config.timeout_per_iteration // 60} minutes")
    
    try:
        start_time = asyncio.get_event_loop().time()
        
        results = await run_cyclical_optimization(
            competition_url="https://realworld.com/loan-default-prediction",
            train_data_path=train_path,
            validation_data_path=val_path,
            config=config,
            output_dir="./real_world_optimization_results"
        )
        
        end_time = asyncio.get_event_loop().time()
        
        # Comprehensive results analysis
        summary = results['optimization_summary']
        
        print(f"\n🎉 Real-world optimization completed!")
        print(f"🏆 Final AUC: {summary['best_performance']:.4f}")
        print(f"🎯 Target AUC: {config.absolute_performance_threshold}")
        print(f"✅ Target achieved: {'Yes' if summary['best_performance'] >= config.absolute_performance_threshold else 'No'}")
        print(f"🔄 Iterations: {summary['total_iterations']}")
        print(f"⏱️ Total time: {(end_time - start_time) / 60:.1f} minutes")
        
        # Performance analysis
        perf_progression = results['performance_progression']
        improvement_progression = results['improvement_progression']
        
        print(f"\n📈 Optimization Journey:")
        best_so_far = 0.0
        for i, (perf, improvement) in enumerate(zip(perf_progression, improvement_progression), 1):
            if perf > best_so_far:
                best_so_far = perf
                status = "🌟 NEW BEST"
            elif improvement > 0:
                status = "📈 Improved"
            else:
                status = "📉 Declined"
            
            print(f"  Iteration {i}: AUC={perf:.4f} ({improvement:+.4f}) {status}")
        
        # Final model configuration
        final_config = results['final_configuration']
        print(f"\n🛠️ Best Configuration Found:")
        print(f"  Models enabled: {list(k for k, v in final_config.get('models', {}).items() if v)}")
        print(f"  Feature engineering: {final_config.get('feature_engineering', False)}")
        print(f"  Feature selection: {final_config.get('feature_selection', False)}")
        print(f"  Ensemble methods: {final_config.get('ensemble_methods', [])}")
        
        return results
        
    except Exception as e:
        print(f"❌ Real-world simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Cleanup
        import os
        for file_path in [train_path, val_path, test_path]:
            if os.path.exists(file_path):
                os.remove(file_path)

async def run_all_cyclical_demos():
    """Run all cyclical optimization demonstrations"""
    print("🤖 CYCLICAL MCP OPTIMIZATION SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 70)
    print()
    print("This demonstration showcases the dual MCP server system that")
    print("iteratively optimizes machine learning competition solutions using:")
    print("• MCP Optimizer Server: Generates improvement strategies")
    print("• MCP Evaluator Server: Assesses performance and convergence")
    print("• Cyclical Orchestrator: Manages the optimization loop")
    print("• Customizable Convergence: Flexible stopping criteria")
    print()
    
    demo_results = {}
    
    try:
        # Demo 1: Basic cyclical optimization
        demo_results['basic'] = await demo_basic_cyclical_optimization()
        
        # Demo 2: Advanced with custom thresholds
        demo_results['advanced'] = await demo_advanced_cyclical_optimization()
        
        # Demo 3: Convergence criteria comparison
        demo_results['convergence'] = await demo_custom_convergence_criteria()
        
        # Demo 4: Real-world simulation
        demo_results['real_world'] = await demo_real_world_simulation()
        
        print("\n" + "=" * 70)
        print("🎉 ALL CYCLICAL OPTIMIZATION DEMOS COMPLETED!")
        print("=" * 70)
        
        # Summary of all demos
        print(f"\n📋 DEMO SUMMARY:")
        successful_demos = sum(1 for result in demo_results.values() if result is not None)
        print(f"✅ Successful demos: {successful_demos}/4")
        
        if demo_results.get('basic'):
            basic_perf = demo_results['basic']['optimization_summary']['best_performance']
            print(f"🔹 Basic optimization best performance: {basic_perf:.4f}")
        
        if demo_results.get('advanced'):
            advanced_perf = demo_results['advanced']['optimization_summary']['best_performance']
            print(f"🔹 Advanced optimization best performance: {advanced_perf:.4f}")
        
        if demo_results.get('convergence'):
            convergence_results = demo_results['convergence']
            best_strategy = max(convergence_results, key=lambda x: x.get('best_performance', 0))
            print(f"🔹 Best convergence strategy: {best_strategy['name']}")
        
        if demo_results.get('real_world'):
            real_world_perf = demo_results['real_world']['optimization_summary']['best_performance']
            print(f"🔹 Real-world simulation AUC: {real_world_perf:.4f}")
        
        print(f"\n🚀 The cyclical MCP system is ready for production use!")
        print(f"💡 Key benefits demonstrated:")
        print(f"  • Autonomous iterative improvement")
        print(f"  • Customizable convergence criteria") 
        print(f"  • AI-powered optimization and evaluation")
        print(f"  • Production-ready performance monitoring")
        
    except Exception as e:
        print(f"\n❌ Demo suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    return demo_results

if __name__ == "__main__":
    # Run all demonstrations
    asyncio.run(run_all_cyclical_demos())