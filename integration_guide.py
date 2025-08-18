"""
Integration Guide - Complete Cyclical MCP System
===============================================

This guide demonstrates how all components work together in the
cyclical MCP optimization system with full project status understanding.
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import time

# Import project analysis components
from project_status_analyzer import generate_status_report
from precondition_validator import validate_project_readiness

# Import main competition toolkit components  
from enhanced_competition_toolkit import (
    EnhancedCompetitionFramework,
    autonomous_competition_solution
)
from cyclical_mcp_system import CyclicalConfig, run_cyclical_optimization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegratedCompetitionSolver:
    """Complete integrated competition solving system with project status awareness"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.project_status = None
        self.validation_report = None
        self.ready_for_cyclical = False
        
    async def initialize_system(self):
        """Initialize the system with comprehensive project analysis"""
        
        logger.info("🚀 Initializing Integrated Competition Solver")
        logger.info("=" * 60)
        
        # Step 1: Analyze project status
        logger.info("📊 Step 1: Analyzing project status...")
        self.project_status = generate_status_report(
            project_root=str(self.project_root),
            print_report=False,
            save_json=False
        )
        
        # Step 2: Validate pre-conditions
        logger.info("🔍 Step 2: Validating pre-conditions...")
        self.validation_report = await validate_project_readiness(str(self.project_root))
        
        # Step 3: Assess readiness
        logger.info("🎯 Step 3: Assessing system readiness...")
        self.ready_for_cyclical = self.validation_report.ready_for_cyclical
        
        # Step 4: Report status
        self._report_initialization_status()
        
        return self.ready_for_cyclical
    
    def _report_initialization_status(self):
        """Report initialization status"""
        
        logger.info("📋 INITIALIZATION REPORT")
        logger.info("-" * 25)
        
        # Project health
        health = self.project_status.project_health
        logger.info(f"Project Health Score: {health.health_score:.1f}/100")
        logger.info(f"Readiness Level: {health.readiness_level}")
        
        # Validation results
        validation = self.validation_report
        logger.info(f"Validation Success Rate: {validation.passed_checks}/{validation.total_checks}")
        logger.info(f"Critical Failures: {validation.critical_failures}")
        
        # Readiness assessment
        status_icon = "✅" if self.ready_for_cyclical else "❌"
        logger.info(f"{status_icon} Cyclical MCP Ready: {self.ready_for_cyclical}")
        
        if self.ready_for_cyclical:
            logger.info("🎉 System fully initialized and ready for cyclical optimization!")
        else:
            logger.warning("⚠️ System needs fixes before cyclical optimization")
            for step in validation.next_steps[:3]:
                logger.warning(f"   • {step}")
    
    async def solve_competition(self, 
                              competition_url: str,
                              train_data_path: str,
                              test_data_path: str,
                              sample_submission_path: Optional[str] = None,
                              use_cyclical_optimization: bool = True) -> Dict[str, Any]:
        """Solve competition with full system integration"""
        
        logger.info("🏆 Starting Integrated Competition Solution")
        logger.info("=" * 50)
        
        if not self.ready_for_cyclical and use_cyclical_optimization:
            logger.warning("⚠️ System not ready for cyclical optimization, falling back to standard mode")
            use_cyclical_optimization = False
        
        start_time = time.time()
        
        # Configure cyclical optimization
        cyclical_config = None
        if use_cyclical_optimization:
            cyclical_config = CyclicalConfig(
                max_iterations=6,
                convergence_threshold=0.005,
                consecutive_no_improvement=3,
                absolute_performance_threshold=0.85,
                performance_metric="cv_score",
                timeout_per_iteration=300,  # 5 minutes per iteration
                save_intermediate_results=True
            )
            
            logger.info("🔄 Cyclical optimization enabled:")
            logger.info(f"   Max iterations: {cyclical_config.max_iterations}")
            logger.info(f"   Target performance: {cyclical_config.absolute_performance_threshold}")
            logger.info(f"   Convergence threshold: {cyclical_config.convergence_threshold}")
        
        # Run autonomous solution
        try:
            solution_results = await asyncio.to_thread(
                autonomous_competition_solution,
                competition_url=competition_url,
                train_data_path=train_data_path,
                test_data_path=test_data_path,
                sample_submission_path=sample_submission_path,
                enable_cyclical_optimization=use_cyclical_optimization,
                cyclical_config=cyclical_config,
                output_dir="./integrated_solution_results"
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Enhance results with system status
            solution_results.update({
                "system_status": {
                    "project_health_score": self.project_status.project_health.health_score,
                    "validation_success_rate": f"{self.validation_report.passed_checks}/{self.validation_report.total_checks}",
                    "cyclical_optimization_used": use_cyclical_optimization,
                    "total_solve_time_minutes": total_time / 60
                },
                "integration_metrics": {
                    "modules_analyzed": len(self.project_status.module_analyses),
                    "integration_points": len(self.project_status.integration_points),
                    "dependencies_satisfied": len(self.project_status.dependency_analysis.missing_dependencies) == 0
                }
            })
            
            logger.info("✅ Competition solution completed successfully!")
            self._report_solution_results(solution_results, total_time)
            
            return solution_results
            
        except Exception as e:
            logger.error(f"❌ Competition solution failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "status": "failed",
                "error": str(e),
                "system_status": {
                    "ready_for_cyclical": self.ready_for_cyclical,
                    "project_health": self.project_status.project_health.health_score
                }
            }
    
    def _report_solution_results(self, results: Dict[str, Any], total_time: float):
        """Report solution results"""
        
        logger.info("🎊 SOLUTION RESULTS SUMMARY")
        logger.info("-" * 30)
        
        # Training results
        training_report = results.get('training_report', {})
        logger.info(f"Models trained: {len(training_report.get('models_trained', []))}")
        logger.info(f"Features used: {training_report.get('feature_count', 'N/A')}")
        
        # Cyclical optimization results
        if 'cyclical_optimization_results' in results:
            cyclical_summary = results['cyclical_optimization_results']['summary']
            logger.info(f"🔄 Cyclical optimization:")
            logger.info(f"   Best performance: {cyclical_summary['best_performance']:.4f}")
            logger.info(f"   Iterations: {cyclical_summary['total_iterations']}")
            logger.info(f"   Converged: {cyclical_summary['convergence_achieved']}")
        
        # System integration
        system_status = results.get('system_status', {})
        logger.info(f"📊 System integration:")
        logger.info(f"   Project health: {system_status.get('project_health_score', 'N/A')}/100")
        logger.info(f"   Validation rate: {system_status.get('validation_success_rate', 'N/A')}")
        logger.info(f"   Total time: {total_time / 60:.1f} minutes")
        
        # Files generated
        files = results.get('files_generated', {})
        logger.info(f"📁 Generated files:")
        for file_type, file_path in files.items():
            logger.info(f"   {file_type}: {file_path}")

async def demonstrate_full_integration():
    """Demonstrate the complete integrated system"""
    
    print("🌟 COMPLETE AI COMPETITION TOOLKIT INTEGRATION DEMONSTRATION")
    print("=" * 70)
    print()
    print("This demonstration shows the full cyclical MCP system in action")
    print("with comprehensive project status understanding and validation.")
    print()
    
    # Step 1: Initialize integrated solver
    solver = IntegratedCompetitionSolver()
    
    ready = await solver.initialize_system()
    
    if not ready:
        print("❌ System not ready for cyclical optimization.")
        print("Please address the issues shown above.")
        return None
    
    # Step 2: Create demonstration dataset
    print("\n📊 Creating demonstration competition dataset...")
    
    np.random.seed(2024)
    n_samples = 1500
    
    # Create realistic tabular data for binary classification
    data = {
        'numerical_1': np.random.normal(0, 1, n_samples),
        'numerical_2': np.random.gamma(2, 1, n_samples),
        'numerical_3': np.random.exponential(1, n_samples),
        'categorical_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples, p=[0.5, 0.3, 0.2]),
        'boolean_feature': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    # Create complex target with realistic signal
    target_signal = (
        0.3 * data['numerical_1'] +
        0.2 * np.log(data['numerical_2'] + 1) +
        0.15 * (data['categorical_1'] == 'A').astype(int) +
        0.1 * (data['categorical_2'] == 'X').astype(int) +
        0.25 * data['boolean_feature'] +
        np.random.normal(0, 0.2, n_samples)
    )
    
    # Convert to binary classification
    data['target'] = (target_signal > np.median(target_signal)).astype(int)
    
    df = pd.DataFrame(data)
    
    # Split into train/test
    train_size = int(0.75 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:].drop('target', axis=1)
    
    # Save datasets
    train_path = "integration_demo_train.csv"
    test_path = "integration_demo_test.csv"
    submission_path = "integration_demo_submission.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Create submission format
    submission_format = pd.DataFrame({
        'id': range(len(test_df)),
        'target': 0
    })
    submission_format.to_csv(submission_path, index=False)
    
    print(f"✅ Dataset created:")
    print(f"   Training samples: {len(train_df):,}")
    print(f"   Test samples: {len(test_df):,}")
    print(f"   Features: {len(train_df.columns) - 1}")
    print(f"   Target distribution: {train_df['target'].value_counts().to_dict()}")
    
    # Step 3: Solve competition with full integration
    print(f"\n🚀 Running integrated competition solution...")
    
    try:
        results = await solver.solve_competition(
            competition_url="https://integration-demo.ai/binary-classification-challenge",
            train_data_path=train_path,
            test_data_path=test_path,
            sample_submission_path=submission_path,
            use_cyclical_optimization=True
        )
        
        if results.get('status') != 'failed':
            print("\n🎉 INTEGRATION DEMONSTRATION COMPLETED SUCCESSFULLY!")
            print("=" * 55)
            
            # Show final performance
            if 'cyclical_optimization_results' in results:
                best_perf = results['cyclical_optimization_results']['summary']['best_performance']
                print(f"🏆 Best performance achieved: {best_perf:.4f}")
            
            # Show system utilization
            system_status = results.get('system_status', {})
            print(f"📊 System health utilized: {system_status.get('project_health_score', 'N/A')}/100")
            print(f"⏱️ Total solution time: {system_status.get('total_solve_time_minutes', 'N/A'):.1f} minutes")
            
            print(f"\n🎯 Key achievements:")
            print(f"   ✅ Complete project status analysis")
            print(f"   ✅ Pre-condition validation passed")
            print(f"   ✅ Cyclical MCP optimization enabled")
            print(f"   ✅ Autonomous competition solving")
            print(f"   ✅ Performance optimization achieved")
            
        return results
        
    except Exception as e:
        print(f"❌ Integration demonstration failed: {e}")
        return None
    
    finally:
        # Cleanup
        import os
        cleanup_files = [train_path, test_path, submission_path]
        for file_path in cleanup_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        print("\n🧹 Demo files cleaned up")

async def quick_readiness_check():
    """Quick check of system readiness"""
    
    print("⚡ QUICK READINESS CHECK")
    print("=" * 25)
    
    # Initialize solver
    solver = IntegratedCompetitionSolver()
    ready = await solver.initialize_system()
    
    print(f"\n🎯 READINESS SUMMARY:")
    print(f"Cyclical MCP Ready: {'✅ YES' if ready else '❌ NO'}")
    
    if ready:
        print("🚀 System is ready for production competition use!")
        print()
        print("Next steps:")
        print("1. Run: python integration_guide.py")
        print("2. Or use: autonomous_competition_solution(..., enable_cyclical_optimization=True)")
        print("3. Monitor results in generated reports")
    else:
        print("❌ System needs fixes before cyclical optimization")
        print()
        print("Required actions:")
        for step in solver.validation_report.next_steps[:5]:
            print(f"• {step}")
    
    return ready

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        # Quick readiness check
        ready = asyncio.run(quick_readiness_check())
        sys.exit(0 if ready else 1)
    else:
        # Full integration demonstration
        results = asyncio.run(demonstrate_full_integration())
        
        if results and results.get('status') != 'failed':
            print("\n✨ The complete AI Competition Toolkit with cyclical MCP")
            print("optimization is now fully operational and ready for use!")
        else:
            print("\n⚠️ Integration demonstration encountered issues.")
            print("Please check the logs and resolve any problems before production use.")