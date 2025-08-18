"""
Cyclical MCP System - Iterative optimization using dual MCP servers
==================================================================

This system implements a cyclical optimization loop using two MCP servers:
1. MCP Optimizer: Generates improved code/strategies
2. MCP Evaluator: Evaluates performance and suggests improvements

The cycle continues until convergence criteria are met.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import yaml

# Import competition components
from ai_competition_toolkit import CompetitionFramework
from competition_analyzer import CompetitionInfo, MCPOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationIteration:
    """Single iteration in the optimization cycle"""
    iteration: int
    timestamp: str
    optimizer_input: Dict[str, Any]
    optimizer_output: Dict[str, Any]
    evaluator_input: Dict[str, Any]
    evaluator_output: Dict[str, Any]
    performance_metrics: Dict[str, float]
    improvement_score: float
    convergence_metrics: Dict[str, float]
    
@dataclass
class CyclicalConfig:
    """Configuration for cyclical MCP system"""
    max_iterations: int = 10
    convergence_threshold: float = 0.001
    min_improvement_threshold: float = 0.01
    performance_metric: str = "cv_score"
    timeout_per_iteration: int = 300  # seconds
    parallel_evaluation: bool = True
    save_intermediate_results: bool = True
    
    # Convergence criteria
    consecutive_no_improvement: int = 3
    relative_improvement_threshold: float = 0.005
    absolute_performance_threshold: Optional[float] = None
    
    # MCP server configurations
    optimizer_config: Dict[str, Any] = None
    evaluator_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.optimizer_config is None:
            self.optimizer_config = {
                "model": "claude-3-sonnet",
                "temperature": 0.7,
                "max_tokens": 4000
            }
        
        if self.evaluator_config is None:
            self.evaluator_config = {
                "model": "claude-3-sonnet", 
                "temperature": 0.3,
                "max_tokens": 2000
            }

class MCPServer(ABC):
    """Abstract base class for MCP servers"""
    
    def __init__(self, server_id: str, config: Dict[str, Any]):
        self.server_id = server_id
        self.config = config
        self.call_history = []
        
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request and return response"""
        pass
    
    def log_call(self, request: Dict[str, Any], response: Dict[str, Any]):
        """Log server call for debugging and analysis"""
        self.call_history.append({
            "timestamp": pd.Timestamp.now().isoformat(),
            "request": request,
            "response": response
        })

class MCPOptimizerServer(MCPServer):
    """MCP server for code and strategy optimization"""
    
    def __init__(self, config: Dict[str, Any], api_key: Optional[str] = None):
        super().__init__("optimizer", config)
        self.api_key = api_key
        self.optimization_strategies = [
            "hyperparameter_tuning",
            "feature_engineering", 
            "model_architecture",
            "ensemble_methods",
            "preprocessing_optimization",
            "cross_validation_strategy"
        ]
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization suggestions based on current state"""
        
        current_performance = request.get("current_performance", {})
        competition_info = request.get("competition_info", {})
        previous_iterations = request.get("previous_iterations", [])
        evaluation_feedback = request.get("evaluation_feedback", {})
        
        # Analyze current state and generate optimization prompt
        optimization_prompt = self._create_optimization_prompt(
            current_performance, competition_info, previous_iterations, evaluation_feedback
        )
        
        # Call AI service for optimization suggestions
        optimization_response = await self._call_ai_service(optimization_prompt)
        
        # Parse and structure the response
        structured_response = self._parse_optimization_response(optimization_response)
        
        response = {
            "server_id": self.server_id,
            "timestamp": pd.Timestamp.now().isoformat(),
            "optimization_suggestions": structured_response,
            "confidence_score": self._calculate_confidence(structured_response),
            "estimated_improvement": self._estimate_improvement(structured_response, current_performance)
        }
        
        self.log_call(request, response)
        return response
    
    def _create_optimization_prompt(self, current_performance: Dict[str, Any], 
                                  competition_info: Dict[str, Any],
                                  previous_iterations: List[Dict[str, Any]],
                                  evaluation_feedback: Dict[str, Any]) -> str:
        """Create detailed optimization prompt for AI"""
        
        prompt = f"""
MACHINE LEARNING COMPETITION OPTIMIZATION REQUEST
===============================================

CURRENT STATE ANALYSIS:
Competition: {competition_info.get('title', 'Unknown')}
Problem Type: {competition_info.get('problem_type', 'auto')}
Evaluation Metric: {competition_info.get('evaluation_metric', 'auto')}

Current Performance Metrics:
{json.dumps(current_performance, indent=2)}

ITERATION HISTORY:
{self._format_iteration_history(previous_iterations)}

EVALUATOR FEEDBACK:
{json.dumps(evaluation_feedback, indent=2)}

OPTIMIZATION OBJECTIVES:
1. Improve primary metric: {competition_info.get('evaluation_metric', 'auto')}
2. Maintain model stability and generalization
3. Ensure competition rule compliance
4. Optimize computational efficiency

PLEASE PROVIDE SPECIFIC OPTIMIZATION SUGGESTIONS IN THE FOLLOWING AREAS:

1. HYPERPARAMETER TUNING:
   - Suggest specific parameter ranges for optimization
   - Identify most impactful parameters for current models
   - Recommend adaptive tuning strategies

2. FEATURE ENGINEERING:
   - Propose new feature creation methods
   - Suggest feature selection improvements
   - Recommend domain-specific transformations

3. MODEL ARCHITECTURE:
   - Evaluate current model choices
   - Suggest model additions or replacements
   - Recommend ensemble strategy improvements

4. PREPROCESSING OPTIMIZATION:
   - Analyze current preprocessing pipeline
   - Suggest improvements for data quality
   - Recommend scaling and normalization adjustments

5. CROSS-VALIDATION STRATEGY:
   - Evaluate current CV approach
   - Suggest improvements for robustness
   - Recommend stratification adjustments

6. IMPLEMENTATION PRIORITIES:
   - Rank suggestions by expected impact
   - Estimate implementation difficulty
   - Suggest quick wins vs. long-term improvements

Please format your response as structured JSON with specific, actionable recommendations.
"""
        
        return prompt
    
    def _format_iteration_history(self, iterations: List[Dict[str, Any]]) -> str:
        """Format iteration history for prompt"""
        if not iterations:
            return "No previous iterations"
        
        history = []
        for i, iteration in enumerate(iterations[-3:]):  # Last 3 iterations
            perf = iteration.get('performance_metrics', {})
            improvements = iteration.get('improvement_score', 0)
            history.append(f"Iteration {iteration.get('iteration', i)}: "
                         f"Score={perf.get('cv_score', 'N/A')}, "
                         f"Improvement={improvements:.4f}")
        
        return "\n".join(history)
    
    async def _call_ai_service(self, prompt: str) -> str:
        """Call AI service for optimization suggestions"""
        try:
            # Use appropriate AI service based on configuration
            if self.config.get("model", "").startswith("claude"):
                return await self._call_anthropic_api(prompt)
            elif self.config.get("model", "").startswith("gpt"):
                return await self._call_openai_api(prompt)
            else:
                return self._rule_based_optimization(prompt)
        
        except Exception as e:
            logger.error(f"AI service call failed: {e}")
            return self._rule_based_optimization(prompt)
    
    async def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API asynchronously"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model=self.config.get("model", "claude-3-sonnet-20240229"),
                max_tokens=self.config.get("max_tokens", 4000),
                temperature=self.config.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return self._rule_based_optimization(prompt)
    
    async def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API asynchronously"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.config.get("model", "gpt-4"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.get("max_tokens", 4000),
                temperature=self.config.get("temperature", 0.7)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return self._rule_based_optimization(prompt)
    
    def _rule_based_optimization(self, prompt: str) -> str:
        """Fallback rule-based optimization"""
        return json.dumps({
            "hyperparameter_tuning": {
                "suggestions": ["Increase n_estimators", "Tune learning_rate", "Optimize max_depth"],
                "priority": "high",
                "expected_improvement": 0.02
            },
            "feature_engineering": {
                "suggestions": ["Add polynomial features", "Create interaction terms", "Apply target encoding"],
                "priority": "medium", 
                "expected_improvement": 0.015
            },
            "model_architecture": {
                "suggestions": ["Add LightGBM model", "Implement stacking ensemble", "Try CatBoost"],
                "priority": "high",
                "expected_improvement": 0.025
            },
            "preprocessing_optimization": {
                "suggestions": ["Robust scaling", "Advanced imputation", "Outlier removal"],
                "priority": "low",
                "expected_improvement": 0.01
            }
        })
    
    def _parse_optimization_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured format"""
        try:
            # Try to parse as JSON first
            return json.loads(response)
        except:
            # Fallback: extract structured information from text
            return {
                "hyperparameter_tuning": {"suggestions": ["Tune key parameters"], "priority": "medium"},
                "feature_engineering": {"suggestions": ["Improve features"], "priority": "medium"},
                "model_architecture": {"suggestions": ["Optimize models"], "priority": "medium"},
                "raw_response": response
            }
    
    def _calculate_confidence(self, suggestions: Dict[str, Any]) -> float:
        """Calculate confidence score for suggestions"""
        # Simple heuristic based on specificity and detail
        total_suggestions = sum(len(category.get("suggestions", [])) 
                              for category in suggestions.values() 
                              if isinstance(category, dict))
        
        return min(0.9, 0.5 + (total_suggestions * 0.05))
    
    def _estimate_improvement(self, suggestions: Dict[str, Any], 
                            current_performance: Dict[str, Any]) -> float:
        """Estimate potential improvement from suggestions"""
        total_improvement = 0.0
        
        for category, details in suggestions.items():
            if isinstance(details, dict) and "expected_improvement" in details:
                total_improvement += details["expected_improvement"]
        
        # Cap estimated improvement at reasonable level
        return min(0.1, total_improvement)

class MCPEvaluatorServer(MCPServer):
    """MCP server for performance evaluation and feedback"""
    
    def __init__(self, config: Dict[str, Any], api_key: Optional[str] = None):
        super().__init__("evaluator", config)
        self.api_key = api_key
        self.evaluation_criteria = [
            "performance_metrics",
            "model_stability", 
            "generalization_ability",
            "computational_efficiency",
            "implementation_quality",
            "competition_compliance"
        ]
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate current solution and provide improvement feedback"""
        
        performance_results = request.get("performance_results", {})
        model_details = request.get("model_details", {})
        optimization_history = request.get("optimization_history", [])
        competition_requirements = request.get("competition_requirements", {})
        
        # Create evaluation prompt
        evaluation_prompt = self._create_evaluation_prompt(
            performance_results, model_details, optimization_history, competition_requirements
        )
        
        # Call AI service for evaluation
        evaluation_response = await self._call_ai_service(evaluation_prompt)
        
        # Parse and analyze the evaluation
        structured_evaluation = self._parse_evaluation_response(evaluation_response)
        
        response = {
            "server_id": self.server_id,
            "timestamp": pd.Timestamp.now().isoformat(),
            "evaluation_results": structured_evaluation,
            "improvement_recommendations": self._generate_recommendations(structured_evaluation),
            "convergence_assessment": self._assess_convergence(optimization_history),
            "overall_score": self._calculate_overall_score(structured_evaluation)
        }
        
        self.log_call(request, response)
        return response
    
    def _create_evaluation_prompt(self, performance_results: Dict[str, Any],
                                model_details: Dict[str, Any],
                                optimization_history: List[Dict[str, Any]],
                                competition_requirements: Dict[str, Any]) -> str:
        """Create comprehensive evaluation prompt"""
        
        prompt = f"""
MACHINE LEARNING SOLUTION EVALUATION REQUEST
==========================================

COMPETITION CONTEXT:
{json.dumps(competition_requirements, indent=2)}

CURRENT PERFORMANCE RESULTS:
{json.dumps(performance_results, indent=2)}

MODEL ARCHITECTURE DETAILS:
{json.dumps(model_details, indent=2)}

OPTIMIZATION HISTORY:
{self._format_optimization_history(optimization_history)}

EVALUATION REQUIREMENTS:

Please provide a comprehensive evaluation across the following dimensions:

1. PERFORMANCE METRICS ANALYSIS:
   - Evaluate current scores against competition benchmarks
   - Assess metric stability across cross-validation folds
   - Identify potential overfitting or underfitting issues
   - Compare performance trends across iterations

2. MODEL QUALITY ASSESSMENT:
   - Evaluate model complexity and interpretability
   - Assess ensemble diversity and effectiveness
   - Review feature importance and selection quality
   - Analyze hyperparameter optimization effectiveness

3. GENERALIZATION CAPABILITY:
   - Evaluate cross-validation consistency
   - Assess potential data leakage issues
   - Review train-validation performance gaps
   - Analyze robustness to data variations

4. COMPUTATIONAL EFFICIENCY:
   - Evaluate training and inference time
   - Assess memory usage and scalability
   - Review optimization convergence rate
   - Analyze resource utilization efficiency

5. IMPLEMENTATION QUALITY:
   - Review code structure and maintainability
   - Assess error handling and robustness
   - Evaluate reproducibility and determinism
   - Check competition rule compliance

6. IMPROVEMENT OPPORTUNITIES:
   - Identify specific bottlenecks and limitations
   - Suggest targeted optimization areas
   - Prioritize improvement recommendations
   - Estimate potential performance gains

7. CONVERGENCE ANALYSIS:
   - Assess optimization progress and trends
   - Evaluate diminishing returns patterns
   - Recommend continuation or termination
   - Suggest alternative optimization strategies

Please provide detailed analysis with specific scores (0-100) for each dimension 
and actionable recommendations for improvement.
"""
        
        return prompt
    
    def _format_optimization_history(self, history: List[Dict[str, Any]]) -> str:
        """Format optimization history for evaluation"""
        if not history:
            return "No optimization history available"
        
        formatted = []
        for iteration in history:
            perf = iteration.get('performance_metrics', {})
            improvement = iteration.get('improvement_score', 0)
            formatted.append(f"Iteration {iteration.get('iteration', 0)}: "
                           f"CV Score: {perf.get('cv_score', 'N/A'):.4f}, "
                           f"Train Score: {perf.get('train_score', 'N/A'):.4f}, "
                           f"Improvement: {improvement:.4f}")
        
        return "\n".join(formatted)
    
    async def _call_ai_service(self, prompt: str) -> str:
        """Call AI service for evaluation"""
        try:
            if self.config.get("model", "").startswith("claude"):
                return await self._call_anthropic_api(prompt)
            elif self.config.get("model", "").startswith("gpt"):
                return await self._call_openai_api(prompt)
            else:
                return self._rule_based_evaluation(prompt)
        
        except Exception as e:
            logger.error(f"AI service call failed: {e}")
            return self._rule_based_evaluation(prompt)
    
    async def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API for evaluation"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model=self.config.get("model", "claude-3-sonnet-20240229"),
                max_tokens=self.config.get("max_tokens", 2000),
                temperature=self.config.get("temperature", 0.3),
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return self._rule_based_evaluation(prompt)
    
    async def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API for evaluation"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.config.get("model", "gpt-4"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.get("max_tokens", 2000),
                temperature=self.config.get("temperature", 0.3)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return self._rule_based_evaluation(prompt)
    
    def _rule_based_evaluation(self, prompt: str) -> str:
        """Fallback rule-based evaluation"""
        return json.dumps({
            "performance_metrics": {"score": 75, "analysis": "Good performance with room for improvement"},
            "model_quality": {"score": 80, "analysis": "Solid model architecture"},
            "generalization": {"score": 70, "analysis": "Some overfitting detected"},
            "efficiency": {"score": 85, "analysis": "Efficient implementation"},
            "implementation": {"score": 90, "analysis": "Clean, well-structured code"},
            "overall_assessment": "Strong solution with optimization opportunities",
            "key_recommendations": [
                "Improve regularization to reduce overfitting",
                "Add more diverse models to ensemble",
                "Optimize feature engineering pipeline"
            ]
        })
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse evaluation response into structured format"""
        try:
            return json.loads(response)
        except:
            # Fallback parsing
            return {
                "performance_metrics": {"score": 75},
                "model_quality": {"score": 75},
                "generalization": {"score": 75},
                "efficiency": {"score": 75},
                "implementation": {"score": 75},
                "raw_response": response
            }
    
    def _generate_recommendations(self, evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific improvement recommendations"""
        recommendations = []
        
        for criterion, details in evaluation.items():
            if isinstance(details, dict) and "score" in details:
                score = details["score"]
                if score < 80:  # Areas needing improvement
                    recommendations.append({
                        "area": criterion,
                        "current_score": score,
                        "priority": "high" if score < 70 else "medium",
                        "suggestions": details.get("recommendations", [])
                    })
        
        return recommendations
    
    def _assess_convergence(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess convergence based on optimization history"""
        if len(history) < 2:
            return {"converged": False, "reason": "Insufficient history"}
        
        # Analyze performance trends
        scores = [h.get('performance_metrics', {}).get('cv_score', 0) for h in history[-5:]]
        improvements = [h.get('improvement_score', 0) for h in history[-3:]]
        
        # Check for convergence patterns
        recent_improvements = [imp for imp in improvements if imp > 0.001]
        
        converged = len(recent_improvements) == 0 and len(history) >= 3
        
        return {
            "converged": converged,
            "recent_scores": scores,
            "recent_improvements": improvements,
            "trend": "declining" if len(recent_improvements) == 0 else "improving",
            "recommendation": "stop" if converged else "continue"
        }
    
    def _calculate_overall_score(self, evaluation: Dict[str, Any]) -> float:
        """Calculate weighted overall score"""
        weights = {
            "performance_metrics": 0.4,
            "model_quality": 0.2,
            "generalization": 0.2,
            "efficiency": 0.1,
            "implementation": 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for criterion, weight in weights.items():
            if criterion in evaluation and isinstance(evaluation[criterion], dict):
                score = evaluation[criterion].get("score", 0)
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

class CyclicalMCPOrchestrator:
    """Orchestrates the cyclical optimization process between MCP servers"""
    
    def __init__(self, config: CyclicalConfig, 
                 optimizer_api_key: Optional[str] = None,
                 evaluator_api_key: Optional[str] = None):
        self.config = config
        self.optimizer_server = MCPOptimizerServer(config.optimizer_config, optimizer_api_key)
        self.evaluator_server = MCPEvaluatorServer(config.evaluator_config, evaluator_api_key)
        
        self.iteration_history: List[OptimizationIteration] = []
        self.current_best_performance = 0.0
        self.no_improvement_count = 0
        self.start_time = None
        
    async def optimize_cyclically(self, 
                                competition_info: CompetitionInfo,
                                initial_framework: CompetitionFramework,
                                train_data: pd.DataFrame,
                                validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run cyclical optimization process"""
        
        logger.info("🔄 Starting cyclical MCP optimization process")
        self.start_time = time.time()
        
        current_framework = initial_framework
        best_framework = initial_framework
        best_performance = 0.0
        
        for iteration in range(self.config.max_iterations):
            logger.info(f"🔄 Starting iteration {iteration + 1}/{self.config.max_iterations}")
            
            try:
                # Run single iteration
                iteration_result = await self._run_single_iteration(
                    iteration + 1, competition_info, current_framework, 
                    train_data, validation_data
                )
                
                # Update best solution if improved
                current_performance = iteration_result.performance_metrics.get(self.config.performance_metric, 0)
                
                if current_performance > best_performance:
                    best_performance = current_performance
                    best_framework = current_framework
                    self.no_improvement_count = 0
                    logger.info(f"✅ New best performance: {best_performance:.4f}")
                else:
                    self.no_improvement_count += 1
                    logger.info(f"⚠️ No improvement for {self.no_improvement_count} iterations")
                
                # Store iteration results
                self.iteration_history.append(iteration_result)
                
                # Check convergence criteria
                should_stop, stop_reason = self._check_convergence(iteration_result)
                if should_stop:
                    logger.info(f"🛑 Stopping optimization: {stop_reason}")
                    break
                
                # Apply optimizations for next iteration
                current_framework = await self._apply_optimizations(
                    current_framework, iteration_result.optimizer_output
                )
                
            except Exception as e:
                logger.error(f"❌ Iteration {iteration + 1} failed: {e}")
                break
        
        # Compile final results
        final_results = self._compile_final_results(best_framework, best_performance)
        
        logger.info(f"🎉 Cyclical optimization completed in {len(self.iteration_history)} iterations")
        logger.info(f"🏆 Best performance: {best_performance:.4f}")
        
        return final_results
    
    async def _run_single_iteration(self, iteration: int, 
                                  competition_info: CompetitionInfo,
                                  framework: CompetitionFramework,
                                  train_data: pd.DataFrame,
                                  validation_data: Optional[pd.DataFrame]) -> OptimizationIteration:
        """Run a single optimization iteration"""
        
        iteration_start = time.time()
        
        # Step 1: Evaluate current solution
        logger.info(f"📊 Evaluating current solution (iteration {iteration})")
        performance_results = await self._evaluate_current_solution(framework, train_data, validation_data)
        
        # Step 2: Get evaluation feedback
        evaluator_request = {
            "performance_results": performance_results,
            "model_details": self._extract_model_details(framework),
            "optimization_history": [asdict(h) for h in self.iteration_history],
            "competition_requirements": self._extract_competition_requirements(competition_info)
        }
        
        evaluator_response = await self.evaluator_server.process_request(evaluator_request)
        
        # Step 3: Get optimization suggestions
        logger.info(f"🤖 Generating optimization suggestions (iteration {iteration})")
        optimizer_request = {
            "current_performance": performance_results,
            "competition_info": self._extract_competition_requirements(competition_info),
            "previous_iterations": [asdict(h) for h in self.iteration_history],
            "evaluation_feedback": evaluator_response["evaluation_results"]
        }
        
        optimizer_response = await self.optimizer_server.process_request(optimizer_request)
        
        # Step 4: Calculate improvement metrics
        improvement_score = self._calculate_improvement_score(performance_results)
        convergence_metrics = self._calculate_convergence_metrics(evaluator_response)
        
        # Create iteration object
        iteration_obj = OptimizationIteration(
            iteration=iteration,
            timestamp=pd.Timestamp.now().isoformat(),
            optimizer_input=optimizer_request,
            optimizer_output=optimizer_response,
            evaluator_input=evaluator_request,
            evaluator_output=evaluator_response,
            performance_metrics=performance_results,
            improvement_score=improvement_score,
            convergence_metrics=convergence_metrics
        )
        
        iteration_time = time.time() - iteration_start
        logger.info(f"⏱️ Iteration {iteration} completed in {iteration_time:.1f}s")
        
        return iteration_obj
    
    async def _evaluate_current_solution(self, framework: CompetitionFramework,
                                       train_data: pd.DataFrame,
                                       validation_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Evaluate current solution performance"""
        
        # Prepare data
        target_col = framework.config.get('target_column', 'target')
        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]
        
        if validation_data is not None:
            X_val = validation_data.drop(columns=[target_col])
            y_val = validation_data[target_col]
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Process data through framework pipeline
        X_train_processed = framework.preprocessor.transform(X_train) if framework.preprocessor.is_fitted else X_train
        X_val_processed = framework.preprocessor.transform(X_val) if framework.preprocessor.is_fitted else X_val
        
        # Evaluate models
        results = {}
        
        # Cross-validation score
        from sklearn.model_selection import cross_val_score
        if framework.trained_models:
            cv_scores = []
            for name, model in framework.trained_models.items():
                try:
                    scores = cross_val_score(model, X_train_processed, y_train, 
                                           cv=framework.config.get('cv_folds', 5), 
                                           scoring=framework.metric)
                    cv_scores.append(scores.mean())
                except:
                    pass
            
            results['cv_score'] = np.mean(cv_scores) if cv_scores else 0.0
            results['cv_std'] = np.std(cv_scores) if cv_scores else 0.0
        
        # Validation score
        if framework.trained_models:
            val_scores = []
            for name, model in framework.trained_models.items():
                try:
                    val_pred = model.predict(X_val_processed)
                    val_score = framework._calculate_score(y_val, val_pred)
                    val_scores.append(val_score)
                except:
                    pass
            
            results['validation_score'] = np.mean(val_scores) if val_scores else 0.0
            results['validation_std'] = np.std(val_scores) if val_scores else 0.0
        
        # Training score
        if framework.trained_models:
            train_scores = []
            for name, model in framework.trained_models.items():
                try:
                    train_pred = model.predict(X_train_processed)
                    train_score = framework._calculate_score(y_train, train_pred)
                    train_scores.append(train_score)
                except:
                    pass
            
            results['train_score'] = np.mean(train_scores) if train_scores else 0.0
        
        # Model diversity (for ensembles)
        if len(framework.trained_models) > 1:
            results['model_diversity'] = self._calculate_model_diversity(framework, X_val_processed)
        
        return results
    
    def _extract_model_details(self, framework: CompetitionFramework) -> Dict[str, Any]:
        """Extract model architecture details"""
        return {
            "models_used": list(framework.trained_models.keys()),
            "ensemble_methods": list(framework.ensemble_manager.ensemble_models.keys()),
            "feature_count": len(framework.feature_engineer.feature_names) if framework.feature_engineer.feature_names else 0,
            "preprocessing_steps": framework.config.get('preprocessing', {}),
            "hyperparameter_optimization": {
                "max_trials": framework.config.get('max_trials', 0),
                "cv_folds": framework.config.get('cv_folds', 0)
            }
        }
    
    def _extract_competition_requirements(self, competition_info: CompetitionInfo) -> Dict[str, Any]:
        """Extract competition requirements"""
        return {
            "title": competition_info.title,
            "problem_type": competition_info.problem_type,
            "evaluation_metric": competition_info.evaluation_metric,
            "target_column": competition_info.target_column,
            "rules": competition_info.rules[:5]  # First 5 rules
        }
    
    def _calculate_improvement_score(self, current_performance: Dict[str, Any]) -> float:
        """Calculate improvement score compared to previous iteration"""
        if not self.iteration_history:
            return 0.0
        
        current_score = current_performance.get(self.config.performance_metric, 0)
        previous_score = self.iteration_history[-1].performance_metrics.get(self.config.performance_metric, 0)
        
        return current_score - previous_score
    
    def _calculate_convergence_metrics(self, evaluator_response: Dict[str, Any]) -> Dict[str, float]:
        """Calculate convergence-related metrics"""
        convergence_assessment = evaluator_response.get("convergence_assessment", {})
        
        return {
            "overall_score": evaluator_response.get("overall_score", 0.0),
            "improvement_trend": 1.0 if convergence_assessment.get("trend") == "improving" else 0.0,
            "convergence_indicator": 1.0 if convergence_assessment.get("converged", False) else 0.0
        }
    
    def _check_convergence(self, iteration_result: OptimizationIteration) -> Tuple[bool, str]:
        """Check if optimization should stop based on convergence criteria"""
        
        # Check maximum iterations
        if iteration_result.iteration >= self.config.max_iterations:
            return True, f"Maximum iterations ({self.config.max_iterations}) reached"
        
        # Check consecutive no improvement
        if self.no_improvement_count >= self.config.consecutive_no_improvement:
            return True, f"No improvement for {self.no_improvement_count} consecutive iterations"
        
        # Check improvement threshold
        if abs(iteration_result.improvement_score) < self.config.convergence_threshold:
            return True, f"Improvement below threshold ({self.config.convergence_threshold})"
        
        # Check absolute performance threshold
        if self.config.absolute_performance_threshold:
            current_performance = iteration_result.performance_metrics.get(self.config.performance_metric, 0)
            if current_performance >= self.config.absolute_performance_threshold:
                return True, f"Absolute performance threshold ({self.config.absolute_performance_threshold}) reached"
        
        # Check evaluator recommendation
        convergence_assessment = iteration_result.evaluator_output.get("convergence_assessment", {})
        if convergence_assessment.get("recommendation") == "stop":
            return True, "Evaluator recommends stopping optimization"
        
        return False, ""
    
    async def _apply_optimizations(self, framework: CompetitionFramework, 
                                 optimizer_output: Dict[str, Any]) -> CompetitionFramework:
        """Apply optimization suggestions to framework"""
        
        suggestions = optimizer_output.get("optimization_suggestions", {})
        
        # Apply configuration optimizations
        for category, details in suggestions.items():
            if category == "hyperparameter_tuning":
                # Increase optimization trials if suggested
                if "increase_trials" in str(details):
                    current_trials = framework.config.get('max_trials', 100)
                    framework.config.set('max_trials', min(200, int(current_trials * 1.5)))
            
            elif category == "model_architecture":
                # Enable additional models if suggested
                suggestions_text = str(details.get("suggestions", []))
                if "lightgbm" in suggestions_text.lower() or "lgb" in suggestions_text.lower():
                    models = framework.config.get('models', {})
                    models['lgb'] = True
                    framework.config.set('models', models)
                
                if "catboost" in suggestions_text.lower():
                    models = framework.config.get('models', {})
                    models['catboost'] = True
                    framework.config.set('models', models)
            
            elif category == "preprocessing_optimization":
                # Apply preprocessing improvements
                preprocessing = framework.config.get('preprocessing', {})
                suggestions_text = str(details.get("suggestions", []))
                
                if "robust" in suggestions_text.lower():
                    preprocessing['scale_features'] = True
                if "outlier" in suggestions_text.lower():
                    preprocessing['remove_outliers'] = True
                
                framework.config.set('preprocessing', preprocessing)
        
        return framework
    
    def _calculate_model_diversity(self, framework: CompetitionFramework, X_val: pd.DataFrame) -> float:
        """Calculate diversity among ensemble models"""
        if len(framework.trained_models) < 2:
            return 0.0
        
        predictions = []
        for model in framework.trained_models.values():
            try:
                pred = model.predict(X_val)
                predictions.append(pred)
            except:
                pass
        
        if len(predictions) < 2:
            return 0.0
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        # Diversity is inverse of average correlation
        avg_correlation = np.mean(correlations) if correlations else 1.0
        return 1.0 - avg_correlation
    
    def _compile_final_results(self, best_framework: CompetitionFramework, 
                             best_performance: float) -> Dict[str, Any]:
        """Compile final optimization results"""
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        results = {
            "optimization_summary": {
                "total_iterations": len(self.iteration_history),
                "total_time_seconds": total_time,
                "best_performance": best_performance,
                "performance_metric": self.config.performance_metric,
                "convergence_achieved": self.no_improvement_count >= self.config.consecutive_no_improvement
            },
            "best_framework": best_framework,
            "iteration_history": [asdict(iteration) for iteration in self.iteration_history],
            "performance_progression": [
                iteration.performance_metrics.get(self.config.performance_metric, 0) 
                for iteration in self.iteration_history
            ],
            "improvement_progression": [
                iteration.improvement_score for iteration in self.iteration_history
            ],
            "final_configuration": best_framework.config.config,
            "mcp_server_stats": {
                "optimizer_calls": len(self.optimizer_server.call_history),
                "evaluator_calls": len(self.evaluator_server.call_history)
            }
        }
        
        return results
    
    def save_optimization_results(self, results: Dict[str, Any], output_dir: str = "./cyclical_optimization_results"):
        """Save optimization results to files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save main results
        with open(output_path / "optimization_results.json", 'w') as f:
            # Remove framework object for JSON serialization
            results_copy = results.copy()
            del results_copy['best_framework']
            json.dump(results_copy, f, indent=2, default=str)
        
        # Save best framework
        results['best_framework'].save_model(str(output_path / "best_framework"))
        
        # Save configuration
        with open(output_path / "optimization_config.yaml", 'w') as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False)
        
        # Save iteration details
        iterations_df = pd.DataFrame([asdict(iteration) for iteration in self.iteration_history])
        iterations_df.to_csv(output_path / "iteration_history.csv", index=False)
        
        logger.info(f"💾 Optimization results saved to {output_dir}")

# Convenience function for easy usage
async def run_cyclical_optimization(competition_url: str,
                                  train_data_path: str,
                                  validation_data_path: Optional[str] = None,
                                  config: Optional[CyclicalConfig] = None,
                                  optimizer_api_key: Optional[str] = None,
                                  evaluator_api_key: Optional[str] = None,
                                  output_dir: str = "./cyclical_optimization_results") -> Dict[str, Any]:
    """
    Run complete cyclical optimization process
    
    Args:
        competition_url: URL of the competition
        train_data_path: Path to training data CSV
        validation_data_path: Path to validation data CSV (optional)
        config: Cyclical optimization configuration (optional)
        optimizer_api_key: API key for optimizer MCP server (optional)
        evaluator_api_key: API key for evaluator MCP server (optional)
        output_dir: Directory to save results
    
    Returns:
        Complete optimization results
    """
    
    # Load data
    train_data = pd.read_csv(train_data_path)
    validation_data = pd.read_csv(validation_data_path) if validation_data_path else None
    
    # Create default config if not provided
    if config is None:
        config = CyclicalConfig(
            max_iterations=10,
            convergence_threshold=0.001,
            min_improvement_threshold=0.01
        )
    
    # Initialize competition analysis
    from enhanced_competition_toolkit import EnhancedCompetitionFramework
    
    initial_framework = EnhancedCompetitionFramework(
        competition_url=competition_url,
        auto_analyze=True
    )
    
    # Prepare initial training
    target_col = initial_framework.config.get('target_column', 'target')
    X_train, y_train, _ = initial_framework.prepare_data(train_data, target_col)
    initial_framework.train_models(X_train, y_train)
    initial_framework.create_ensembles(X_train, y_train)
    
    # Create competition info
    competition_info = CompetitionInfo(
        title=competition_url.split('/')[-1],
        description="",
        problem_type=initial_framework.problem_type or "auto",
        evaluation_metric=initial_framework.metric or "auto",
        submission_format="csv",
        deadline="",
        rules=[],
        data_description="",
        target_column=target_col,
        sample_submission_format={}
    )
    
    # Run cyclical optimization
    orchestrator = CyclicalMCPOrchestrator(config, optimizer_api_key, evaluator_api_key)
    
    results = await orchestrator.optimize_cyclically(
        competition_info, initial_framework, train_data, validation_data
    )
    
    # Save results
    orchestrator.save_optimization_results(results, output_dir)
    
    return results

if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def demo_cyclical_optimization():
        """Demonstrate cyclical optimization"""
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.randint(0, 2, n_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_csv('demo_cyclical_train.csv', index=False)
        
        # Run cyclical optimization
        config = CyclicalConfig(
            max_iterations=3,  # Small for demo
            convergence_threshold=0.01,
            consecutive_no_improvement=2
        )
        
        try:
            results = await run_cyclical_optimization(
                competition_url="https://example.com/demo-competition",
                train_data_path="demo_cyclical_train.csv",
                config=config,
                output_dir="./demo_cyclical_results"
            )
            
            print("🎉 Cyclical optimization completed!")
            print(f"📊 Best performance: {results['optimization_summary']['best_performance']:.4f}")
            print(f"🔄 Total iterations: {results['optimization_summary']['total_iterations']}")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
        
        finally:
            # Cleanup
            import os
            if os.path.exists('demo_cyclical_train.csv'):
                os.remove('demo_cyclical_train.csv')
    
    # Run demo
    asyncio.run(demo_cyclical_optimization())