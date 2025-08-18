"""
Competition Analyzer - Automated competition requirement analysis and code optimization
"""

import requests
import re
import json
import ast
import os
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from github import Github
import git
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CompetitionInfo:
    """Competition information structure"""
    title: str
    description: str
    problem_type: str
    evaluation_metric: str
    submission_format: str
    deadline: str
    rules: List[str]
    data_description: str
    target_column: str
    sample_submission_format: Dict[str, Any]
    
class CompetitionScraper:
    """Web scraper for competition platforms (Kaggle, DrivenData, etc.)"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_kaggle_competition(self, competition_url: str) -> CompetitionInfo:
        """Scrape Kaggle competition details"""
        try:
            response = self.session.get(competition_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic information
            title = self._extract_kaggle_title(soup)
            description = self._extract_kaggle_description(soup)
            evaluation_metric = self._extract_kaggle_metric(soup)
            rules = self._extract_kaggle_rules(soup)
            data_description = self._extract_kaggle_data_description(soup)
            
            # Determine problem type from description and metric
            problem_type = self._determine_problem_type(description, evaluation_metric)
            
            # Extract submission format
            submission_format = self._extract_kaggle_submission_format(soup)
            
            return CompetitionInfo(
                title=title,
                description=description,
                problem_type=problem_type,
                evaluation_metric=evaluation_metric,
                submission_format=submission_format,
                deadline=self._extract_kaggle_deadline(soup),
                rules=rules,
                data_description=data_description,
                target_column=self._extract_target_column(data_description, submission_format),
                sample_submission_format={}
            )
            
        except Exception as e:
            logger.error(f"Error scraping Kaggle competition: {e}")
            return self._create_fallback_info()
    
    def scrape_drivendata_competition(self, competition_url: str) -> CompetitionInfo:
        """Scrape DrivenData competition details"""
        try:
            response = self.session.get(competition_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # DrivenData specific parsing
            title = soup.find('h1', class_='competition-title')
            title = title.text.strip() if title else "Unknown Competition"
            
            description = soup.find('div', class_='competition-description')
            description = description.text.strip() if description else ""
            
            # Extract other details...
            
            return CompetitionInfo(
                title=title,
                description=description,
                problem_type=self._determine_problem_type(description, ""),
                evaluation_metric="",
                submission_format="",
                deadline="",
                rules=[],
                data_description="",
                target_column="target",
                sample_submission_format={}
            )
            
        except Exception as e:
            logger.error(f"Error scraping DrivenData competition: {e}")
            return self._create_fallback_info()
    
    def _extract_kaggle_title(self, soup: BeautifulSoup) -> str:
        """Extract competition title from Kaggle page"""
        title_selectors = [
            'h1[data-testid="competition-title"]',
            'h1.competition-title',
            'h1',
            '.competition-header h1'
        ]
        
        for selector in title_selectors:
            title = soup.select_one(selector)
            if title:
                return title.text.strip()
        
        return "Unknown Competition"
    
    def _extract_kaggle_description(self, soup: BeautifulSoup) -> str:
        """Extract competition description"""
        desc_selectors = [
            '[data-testid="competition-description"]',
            '.competition-description',
            '.overview-section'
        ]
        
        for selector in desc_selectors:
            desc = soup.select_one(selector)
            if desc:
                return desc.text.strip()
        
        return ""
    
    def _extract_kaggle_metric(self, soup: BeautifulSoup) -> str:
        """Extract evaluation metric"""
        metric_patterns = [
            r'evaluation.*?metric.*?(\w+)',
            r'scored.*?using.*?(\w+)',
            r'metric.*?(\w+)',
            r'(auc|accuracy|rmse|mae|f1|precision|recall)'
        ]
        
        text = soup.text.lower()
        for pattern in metric_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return "auto"
    
    def _extract_kaggle_rules(self, soup: BeautifulSoup) -> List[str]:
        """Extract competition rules"""
        rules = []
        rules_section = soup.find(text=re.compile("rules", re.I))
        
        if rules_section:
            parent = rules_section.parent
            for i in range(5):  # Look in nearby elements
                if parent:
                    rule_items = parent.find_all(['li', 'p'])
                    for item in rule_items:
                        rule_text = item.text.strip()
                        if len(rule_text) > 10:  # Filter out short text
                            rules.append(rule_text)
                    parent = parent.parent
        
        return rules[:10]  # Limit to first 10 rules
    
    def _extract_kaggle_data_description(self, soup: BeautifulSoup) -> str:
        """Extract data description"""
        data_desc_selectors = [
            '[data-testid="data-description"]',
            '.data-description',
            '#data'
        ]
        
        for selector in data_desc_selectors:
            desc = soup.select_one(selector)
            if desc:
                return desc.text.strip()
        
        return ""
    
    def _extract_kaggle_submission_format(self, soup: BeautifulSoup) -> str:
        """Extract submission format requirements"""
        format_text = soup.text.lower()
        
        if 'csv' in format_text:
            return 'csv'
        elif 'json' in format_text:
            return 'json'
        else:
            return 'csv'  # Default
    
    def _extract_kaggle_deadline(self, soup: BeautifulSoup) -> str:
        """Extract competition deadline"""
        deadline_patterns = [
            r'deadline.*?(\d{4}-\d{2}-\d{2})',
            r'ends.*?(\d{1,2}/\d{1,2}/\d{4})',
            r'closes.*?(\w+ \d{1,2}, \d{4})'
        ]
        
        text = soup.text
        for pattern in deadline_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(1)
        
        return ""
    
    def _extract_target_column(self, data_description: str, submission_format: str) -> str:
        """Extract target column name from descriptions"""
        target_patterns = [
            r'target.*?column.*?(\w+)',
            r'predict.*?(\w+)',
            r'prediction.*?column.*?(\w+)'
        ]
        
        text = (data_description + " " + submission_format).lower()
        for pattern in target_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return "target"  # Default
    
    def _determine_problem_type(self, description: str, metric: str) -> str:
        """Determine if it's classification or regression"""
        classification_indicators = [
            'classification', 'classify', 'predict class', 'category', 'label',
            'auc', 'accuracy', 'f1', 'precision', 'recall'
        ]
        
        regression_indicators = [
            'regression', 'predict value', 'continuous', 'amount', 'price',
            'rmse', 'mae', 'mse', 'r2'
        ]
        
        text = (description + " " + metric).lower()
        
        class_score = sum(1 for indicator in classification_indicators if indicator in text)
        reg_score = sum(1 for indicator in regression_indicators if indicator in text)
        
        if class_score > reg_score:
            return 'classification'
        elif reg_score > class_score:
            return 'regression'
        else:
            return 'auto'
    
    def _create_fallback_info(self) -> CompetitionInfo:
        """Create fallback competition info when scraping fails"""
        return CompetitionInfo(
            title="Unknown Competition",
            description="",
            problem_type="auto",
            evaluation_metric="auto",
            submission_format="csv",
            deadline="",
            rules=[],
            data_description="",
            target_column="target",
            sample_submission_format={}
        )

class GitHubAnalyzer:
    """Analyze GitHub repositories for competition solutions and best practices"""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github = Github(github_token) if github_token else Github()
        self.temp_dir = Path("./temp_repos")
        self.temp_dir.mkdir(exist_ok=True)
    
    def analyze_competition_repos(self, search_query: str, max_repos: int = 10) -> Dict[str, Any]:
        """Search and analyze GitHub repositories for competition solutions"""
        try:
            repos = self.github.search_repositories(
                query=search_query,
                sort="stars",
                order="desc"
            )
            
            analysis_results = {
                'repositories': [],
                'common_patterns': {},
                'best_practices': [],
                'code_snippets': {},
                'model_architectures': [],
                'feature_engineering_techniques': []
            }
            
            count = 0
            for repo in repos:
                if count >= max_repos:
                    break
                
                try:
                    repo_analysis = self._analyze_single_repo(repo)
                    analysis_results['repositories'].append(repo_analysis)
                    count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze repo {repo.name}: {e}")
                    continue
            
            # Extract common patterns
            analysis_results['common_patterns'] = self._extract_common_patterns(
                analysis_results['repositories']
            )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing GitHub repositories: {e}")
            return {'error': str(e)}

    def analyze_specific_repos(self, repo_urls: List[str]) -> Dict[str, Any]:
        """Analyze specific repositories provided via URLs."""
        analysis_results = {
            'repositories': [],
            'common_patterns': {},
            'best_practices': [],
            'code_snippets': {},
            'model_architectures': [],
            'feature_engineering_techniques': []
        }

        for url in repo_urls or []:
            try:
                parsed = urlparse(url)
                parts = [p for p in parsed.path.split('/') if p]
                if len(parts) >= 2:
                    full_name = f"{parts[0]}/{parts[1]}"
                else:
                    logger.warning(f"Invalid GitHub URL: {url}")
                    continue

                try:
                    repo = self.github.get_repo(full_name)
                    repo_analysis = self._analyze_single_repo(repo)
                    analysis_results['repositories'].append(repo_analysis)
                except Exception as e:
                    logger.warning(f"Failed to fetch repo {full_name}: {e}")
                    continue
            except Exception as e:
                logger.warning(f"Could not parse repo URL {url}: {e}")
                continue

        # Derive common patterns
        analysis_results['common_patterns'] = self._extract_common_patterns(
            analysis_results['repositories']
        )

        return analysis_results
    
    def _analyze_single_repo(self, repo) -> Dict[str, Any]:
        """Analyze a single GitHub repository"""
        analysis = {
            'name': repo.name,
            'description': repo.description,
            'stars': repo.stargazers_count,
            'language': repo.language,
            'topics': repo.get_topics(),
            'files': {},
            'models_used': [],
            'techniques': [],
            'performance_metrics': {}
        }
        
        try:
            # Clone repository temporarily
            local_path = self.temp_dir / repo.name
            if local_path.exists():
                import shutil
                shutil.rmtree(local_path)
            
            git.Repo.clone_from(repo.clone_url, local_path, depth=1)
            
            # Analyze code files
            analysis['files'] = self._analyze_code_files(local_path)
            analysis['models_used'] = self._extract_models_used(local_path)
            analysis['techniques'] = self._extract_techniques(local_path)
            
            # Clean up
            import shutil
            shutil.rmtree(local_path)
            
        except Exception as e:
            logger.warning(f"Could not clone repo {repo.name}: {e}")
            
            # Fallback: analyze via GitHub API
            try:
                contents = repo.get_contents("")
                for content in contents:
                    if content.name.endswith(('.py', '.ipynb', '.R')):
                        file_content = content.decoded_content.decode('utf-8')
                        analysis['files'][content.name] = self._analyze_file_content(file_content)
            except Exception as e2:
                logger.warning(f"API analysis also failed for {repo.name}: {e2}")
        
        return analysis
    
    def _analyze_code_files(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze code files in repository"""
        files_analysis = {}
        
        for file_path in repo_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                relative_path = file_path.relative_to(repo_path)
                files_analysis[str(relative_path)] = self._analyze_file_content(content)
                
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")
        
        return files_analysis
    
    def _analyze_file_content(self, content: str) -> Dict[str, Any]:
        """Analyze individual file content"""
        analysis = {
            'imports': [],
            'functions': [],
            'classes': [],
            'models': [],
            'preprocessing_steps': [],
            'feature_engineering': [],
            'evaluation_metrics': []
        }
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        analysis['imports'].append(f"{module}.{alias.name}")
                
                elif isinstance(node, ast.FunctionDef):
                    analysis['functions'].append(node.name)
                
                elif isinstance(node, ast.ClassDef):
                    analysis['classes'].append(node.name)
            
            # Extract specific patterns
            analysis['models'] = self._extract_model_patterns(content)
            analysis['preprocessing_steps'] = self._extract_preprocessing_patterns(content)
            analysis['feature_engineering'] = self._extract_feature_engineering_patterns(content)
            analysis['evaluation_metrics'] = self._extract_metric_patterns(content)
            
        except SyntaxError:
            # Handle notebooks and other formats
            analysis['imports'] = re.findall(r'import\s+(\w+)', content)
            analysis['models'] = self._extract_model_patterns(content)
        
        return analysis
    
    def _extract_model_patterns(self, content: str) -> List[str]:
        """Extract ML model usage patterns"""
        model_patterns = [
            r'(LGBMClassifier|LGBMRegressor)',
            r'(XGBClassifier|XGBRegressor)',
            r'(CatBoostClassifier|CatBoostRegressor)',
            r'(RandomForestClassifier|RandomForestRegressor)',
            r'(LogisticRegression|LinearRegression)',
            r'(SVC|SVR)',
            r'(VotingClassifier|VotingRegressor)',
            r'(StackingClassifier|StackingRegressor)',
            r'(GradientBoostingClassifier|GradientBoostingRegressor)'
        ]
        
        models = []
        for pattern in model_patterns:
            matches = re.findall(pattern, content)
            models.extend(matches)
        
        return list(set(models))
    
    def _extract_preprocessing_patterns(self, content: str) -> List[str]:
        """Extract preprocessing technique patterns"""
        preprocessing_patterns = [
            r'(StandardScaler|MinMaxScaler|RobustScaler)',
            r'(LabelEncoder|OneHotEncoder)',
            r'(fillna|dropna)',
            r'(train_test_split)',
            r'(StratifiedKFold|KFold)',
            r'(SimpleImputer|KNNImputer)'
        ]
        
        techniques = []
        for pattern in preprocessing_patterns:
            matches = re.findall(pattern, content)
            techniques.extend(matches)
        
        return list(set(techniques))
    
    def _extract_feature_engineering_patterns(self, content: str) -> List[str]:
        """Extract feature engineering patterns"""
        fe_patterns = [
            r'(PolynomialFeatures)',
            r'(SelectKBest|SelectFromModel)',
            r'(PCA|TruncatedSVD)',
            r'(FeatureUnion|Pipeline)',
            r'(GroupBy|agg|transform)',
            r'(rolling|expanding|ewm)'
        ]
        
        techniques = []
        for pattern in fe_patterns:
            matches = re.findall(pattern, content)
            techniques.extend(matches)
        
        return list(set(techniques))
    
    def _extract_metric_patterns(self, content: str) -> List[str]:
        """Extract evaluation metric patterns"""
        metric_patterns = [
            r'(roc_auc_score|accuracy_score|f1_score)',
            r'(mean_squared_error|mean_absolute_error)',
            r'(log_loss|precision_score|recall_score)',
            r'(r2_score|explained_variance_score)'
        ]
        
        metrics = []
        for pattern in metric_patterns:
            matches = re.findall(pattern, content)
            metrics.extend(matches)
        
        return list(set(metrics))
    
    def _extract_models_used(self, repo_path: Path) -> List[str]:
        """Extract all models used across the repository"""
        all_models = []
        
        for file_path in repo_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                models = self._extract_model_patterns(content)
                all_models.extend(models)
            except:
                continue
        
        return list(set(all_models))
    
    def _extract_techniques(self, repo_path: Path) -> List[str]:
        """Extract all techniques used across the repository"""
        all_techniques = []
        
        for file_path in repo_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                techniques = []
                techniques.extend(self._extract_preprocessing_patterns(content))
                techniques.extend(self._extract_feature_engineering_patterns(content))
                techniques.extend(self._extract_metric_patterns(content))
                
                all_techniques.extend(techniques)
            except:
                continue
        
        return list(set(all_techniques))
    
    def _extract_common_patterns(self, repositories: List[Dict]) -> Dict[str, Any]:
        """Extract common patterns across all analyzed repositories"""
        all_models = []
        all_techniques = []
        all_imports = []
        
        for repo in repositories:
            all_models.extend(repo.get('models_used', []))
            all_techniques.extend(repo.get('techniques', []))
            
            for file_analysis in repo.get('files', {}).values():
                all_imports.extend(file_analysis.get('imports', []))
        
        # Count frequencies
        from collections import Counter
        
        return {
            'most_common_models': Counter(all_models).most_common(10),
            'most_common_techniques': Counter(all_techniques).most_common(10),
            'most_common_imports': Counter(all_imports).most_common(20)
        }

class MCPOptimizer:
    """Model Context Protocol integration for autonomous code optimization"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet"):
        self.api_key = api_key
        self.model = model
        self.optimization_history = []
    
    def optimize_competition_code(self, 
                                competition_info: CompetitionInfo,
                                github_analysis: Dict[str, Any],
                                current_code: str) -> str:
        """Use MCP to optimize code based on competition requirements and best practices"""
        
        optimization_prompt = self._create_optimization_prompt(
            competition_info, github_analysis, current_code
        )
        
        try:
            optimized_code = self._call_mcp_service(optimization_prompt)
            
            self.optimization_history.append({
                'original_code': current_code,
                'optimized_code': optimized_code,
                'competition': competition_info.title,
                'improvements': self._analyze_improvements(current_code, optimized_code)
            })
            
            return optimized_code
            
        except Exception as e:
            logger.error(f"MCP optimization failed: {e}")
            return current_code
    
    def _create_optimization_prompt(self, 
                                  competition_info: CompetitionInfo,
                                  github_analysis: Dict[str, Any],
                                  current_code: str) -> str:
        """Create optimization prompt for MCP"""
        
        common_patterns = github_analysis.get('common_patterns', {})
        best_models = [model[0] for model in common_patterns.get('most_common_models', [])]
        best_techniques = [tech[0] for tech in common_patterns.get('most_common_techniques', [])]
        
        prompt = f"""
        COMPETITION OPTIMIZATION REQUEST
        
        Competition Details:
        - Title: {competition_info.title}
        - Problem Type: {competition_info.problem_type}
        - Evaluation Metric: {competition_info.evaluation_metric}
        - Target Column: {competition_info.target_column}
        
        Competition Rules:
        {chr(10).join(f"- {rule}" for rule in competition_info.rules[:5])}
        
        Best Practices from Analysis:
        - Most successful models: {best_models[:5]}
        - Common techniques: {best_techniques[:10]}
        
        Current Code to Optimize:
        {current_code}
        
        OPTIMIZATION REQUIREMENTS:
        1. Ensure code follows competition rules and requirements
        2. Incorporate successful patterns from analyzed repositories
        3. Optimize for the specific evaluation metric: {competition_info.evaluation_metric}
        4. Improve feature engineering based on common patterns
        5. Enhance model selection and hyperparameter tuning
        6. Add ensemble methods if beneficial
        7. Ensure robust cross-validation strategy
        8. Optimize for competition-specific submission format
        
        Please provide the optimized code with explanations for key improvements.
        """
        
        return prompt
    
    def _call_mcp_service(self, prompt: str) -> str:
        """Call MCP service for code optimization"""
        # This would integrate with actual MCP services
        # For now, implementing a simplified version
        
        if self.model.startswith("claude"):
            return self._call_anthropic_api(prompt)
        elif self.model.startswith("gpt"):
            return self._call_openai_api(prompt)
        else:
            # Fallback to rule-based optimization
            return self._rule_based_optimization(prompt)
    
    def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API for optimization"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return self._rule_based_optimization(prompt)
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API for optimization"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                max_tokens=4000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return self._rule_based_optimization(prompt)
    
    def _rule_based_optimization(self, prompt: str) -> str:
        """Fallback rule-based optimization"""
        # Extract current code from prompt
        code_start = prompt.find("Current Code to Optimize:") + len("Current Code to Optimize:")
        current_code = prompt[code_start:].strip()
        
        # Apply basic optimization rules
        optimizations = [
            "# OPTIMIZED CODE - Applied rule-based improvements\n",
            "# Added based on competition analysis\n\n",
            current_code,
            "\n\n# Suggested improvements:\n",
            "# 1. Add more cross-validation folds for better stability\n",
            "# 2. Include ensemble methods for improved performance\n",
            "# 3. Add feature engineering based on common patterns\n"
        ]
        
        return "".join(optimizations)
    
    def _analyze_improvements(self, original: str, optimized: str) -> List[str]:
        """Analyze what improvements were made"""
        improvements = []
        
        if len(optimized) > len(original):
            improvements.append("Code expanded with additional functionality")
        
        if "ensemble" in optimized.lower() and "ensemble" not in original.lower():
            improvements.append("Added ensemble methods")
        
        if "cross_val" in optimized and "cross_val" not in original:
            improvements.append("Enhanced cross-validation")
        
        if "feature_engineering" in optimized and "feature_engineering" not in original:
            improvements.append("Improved feature engineering")
        
        return improvements

class CompetitionRequirementsAnalyzer:
    """Analyze competition requirements and generate optimized configuration"""
    
    def __init__(self):
        self.scraper = CompetitionScraper()
        self.github_analyzer = GitHubAnalyzer()
        self.mcp_optimizer = MCPOptimizer()
    
    def analyze_competition(self, 
                          competition_url: str,
                          github_search_query: Optional[str] = None,
                          github_token: Optional[str] = None,
                          mcp_api_key: Optional[str] = None) -> Dict[str, Any]:
        """Complete competition analysis pipeline"""
        
        logger.info("Starting competition analysis...")
        
        # Step 1: Scrape competition requirements
        logger.info("Scraping competition requirements...")
        if "kaggle.com" in competition_url:
            competition_info = self.scraper.scrape_kaggle_competition(competition_url)
        elif "drivendata.org" in competition_url:
            competition_info = self.scraper.scrape_drivendata_competition(competition_url)
        elif "forge.allora.network/competitions/16" in competition_url:
            # Hardcode known details for Allora Forge competition 16
            competition_info = CompetitionInfo(
                title="1 day SOL/USD Log-Return Prediction",
                description="Predict next-day log return for SOL/USD.",
                problem_type="regression",
                evaluation_metric="rmse",
                submission_format="csv",
                deadline="",
                rules=[],
                data_description="Time-series forecasting of 1-day log returns for SOL/USD.",
                target_column="target",
                sample_submission_format={}
            )
        else:
            logger.warning("Unknown competition platform, using fallback")
            competition_info = self.scraper._create_fallback_info()
        
        # Step 2: Analyze GitHub repositories
        logger.info("Analyzing GitHub repositories...")
        if github_search_query is None:
            github_search_query = f"{competition_info.title} machine learning competition"
        
        if github_token:
            self.github_analyzer = GitHubAnalyzer(github_token)
        
        github_analysis = self.github_analyzer.analyze_competition_repos(github_search_query)
        
        # Step 3: Generate optimized configuration
        logger.info("Generating optimized configuration...")
        optimized_config = self._generate_optimized_config(competition_info, github_analysis)
        
        # Step 4: Create code templates
        logger.info("Creating optimized code templates...")
        code_templates = self._create_code_templates(competition_info, github_analysis)
        
        return {
            'competition_info': competition_info,
            'github_analysis': github_analysis,
            'optimized_config': optimized_config,
            'code_templates': code_templates,
            'recommendations': self._generate_recommendations(competition_info, github_analysis)
        }
    
    def _generate_optimized_config(self, 
                                 competition_info: CompetitionInfo,
                                 github_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized configuration based on analysis"""
        
        config = {
            'problem_type': competition_info.problem_type,
            'target_column': competition_info.target_column,
            'metric': competition_info.evaluation_metric,
            'cv_folds': 5,
            'random_state': 42,
            'max_trials': 100
        }
        
        # Optimize based on GitHub analysis
        common_patterns = github_analysis.get('common_patterns', {})
        common_models = [model[0] for model in common_patterns.get('most_common_models', [])]
        
        # Enable models based on popularity in similar competitions
        models_config = {
            'lgb': 'LGBMClassifier' in common_models or 'LGBMRegressor' in common_models,
            'xgb': 'XGBClassifier' in common_models or 'XGBRegressor' in common_models,
            'catboost': 'CatBoostClassifier' in common_models or 'CatBoostRegressor' in common_models,
            'rf': 'RandomForestClassifier' in common_models or 'RandomForestRegressor' in common_models,
            'lr': 'LogisticRegression' in common_models or 'LinearRegression' in common_models,
            'svm': False  # Usually too slow for competitions
        }
        
        config['models'] = models_config
        
        # Optimize preprocessing based on common techniques
        common_techniques = [tech[0] for tech in common_patterns.get('most_common_techniques', [])]
        
        preprocessing_config = {
            'handle_missing': 'fillna' in common_techniques or 'SimpleImputer' in common_techniques,
            'encode_categorical': 'LabelEncoder' in common_techniques or 'OneHotEncoder' in common_techniques,
            'scale_features': any(scaler in common_techniques for scaler in ['StandardScaler', 'MinMaxScaler', 'RobustScaler']),
            'remove_outliers': False  # Conservative default
        }
        
        config['preprocessing'] = preprocessing_config
        
        # Set ensemble methods
        config['ensemble_methods'] = ['voting', 'stacking']
        
        # Feature engineering
        config['feature_engineering'] = 'PolynomialFeatures' in common_techniques
        config['feature_selection'] = 'SelectKBest' in common_techniques or 'SelectFromModel' in common_techniques
        
        return config
    
    def _create_code_templates(self, 
                             competition_info: CompetitionInfo,
                             github_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Create optimized code templates"""
        
        templates = {}
        
        # Main training template
        templates['main_training'] = f"""
# Auto-generated training script for {competition_info.title}
# Optimized based on {len(github_analysis.get('repositories', []))} analyzed repositories

from ai_competition_toolkit import CompetitionFramework
import pandas as pd

def main():
    # Load data
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    # Initialize framework with optimized configuration
    framework = CompetitionFramework()
    
    # Apply competition-specific optimizations
    framework.config.set('problem_type', '{competition_info.problem_type}')
    framework.config.set('target_column', '{competition_info.target_column}')
    framework.config.set('metric', '{competition_info.evaluation_metric}')
    
    # Prepare data
    X_train, y_train, X_test = framework.prepare_data(train_data, '{competition_info.target_column}', test_data)
    
    # Train models
    framework.train_models(X_train, y_train)
    
    # Create ensembles
    framework.create_ensembles(X_train, y_train)
    
    # Generate submission
    submission_format = pd.read_csv('sample_submission.csv')
    framework.generate_submission(X_test, submission_format, 'submission.csv')
    
    print("Training completed! Submission file generated.")

if __name__ == "__main__":
    main()
"""
        
        return templates
    
    def _generate_recommendations(self, 
                                competition_info: CompetitionInfo,
                                github_analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on analysis"""
        
        recommendations = []
        
        # Competition-specific recommendations
        if competition_info.problem_type == 'classification':
            recommendations.append("Focus on classification metrics and ensure proper class balancing")
        elif competition_info.problem_type == 'regression':
            recommendations.append("Pay attention to outliers and consider robust scaling methods")
        
        # GitHub analysis recommendations
        common_patterns = github_analysis.get('common_patterns', {})
        if common_patterns:
            top_models = [model[0] for model in common_patterns.get('most_common_models', [])[:3]]
            if top_models:
                recommendations.append(f"Most successful models in similar competitions: {', '.join(top_models)}")
        
        # Metric-specific recommendations
        if 'auc' in competition_info.evaluation_metric.lower():
            recommendations.append("Focus on probability calibration and ensemble diversity for AUC optimization")
        elif 'rmse' in competition_info.evaluation_metric.lower():
            recommendations.append("Consider robust preprocessing and ensemble methods to minimize RMSE")
        
        return recommendations

# Main usage function
def analyze_competition_automatically(competition_url: str,
                                    github_token: Optional[str] = None,
                                    mcp_api_key: Optional[str] = None,
                                    output_dir: str = "./competition_analysis",
                                    repo_urls: Optional[List[str]] = None) -> Dict[str, Any]:
    """Automatically analyze competition and generate optimized toolkit configuration"""
    
    analyzer = CompetitionRequirementsAnalyzer()
    
    # Set up MCP if API key provided
    if mcp_api_key:
        analyzer.mcp_optimizer = MCPOptimizer(api_key=mcp_api_key)
    
    # Perform analysis
    analysis_results = analyzer.analyze_competition(
        competition_url=competition_url,
        github_token=github_token,
        mcp_api_key=mcp_api_key
    )

    # If specific repositories are provided, analyze and merge them
    if repo_urls:
        try:
            specific = analyzer.github_analyzer.analyze_specific_repos(repo_urls)
            base_repos = analysis_results['github_analysis'].get('repositories', [])
            base_repos.extend(specific.get('repositories', []))
            analysis_results['github_analysis']['repositories'] = base_repos
            # Recompute common patterns across all
            analysis_results['github_analysis']['common_patterns'] = analyzer.github_analyzer._extract_common_patterns(base_repos)
        except Exception as e:
            logger.warning(f"Failed to analyze specific repos: {e}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save configuration
    config_path = output_path / "optimized_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(analysis_results['optimized_config'], f, default_flow_style=False)
    
    # Save code templates
    templates_dir = output_path / "code_templates"
    templates_dir.mkdir(exist_ok=True)
    
    for template_name, template_code in analysis_results['code_templates'].items():
        template_path = templates_dir / f"{template_name}.py"
        with open(template_path, 'w') as f:
            f.write(template_code)
    
    # Save analysis report
    report_path = output_path / "analysis_report.json"
    with open(report_path, 'w') as f:
        # Convert CompetitionInfo to dict for JSON serialization
        report_data = analysis_results.copy()
        report_data['competition_info'] = {
            'title': analysis_results['competition_info'].title,
            'description': analysis_results['competition_info'].description,
            'problem_type': analysis_results['competition_info'].problem_type,
            'evaluation_metric': analysis_results['competition_info'].evaluation_metric,
            'target_column': analysis_results['competition_info'].target_column,
            'rules': analysis_results['competition_info'].rules
        }
        json.dump(report_data, f, indent=2, default=str)
    
    logger.info(f"Analysis complete! Results saved to {output_dir}")
    
    return analysis_results

if __name__ == "__main__":
    # Example usage
    competition_url = "https://www.kaggle.com/competitions/titanic"
    
    results = analyze_competition_automatically(
        competition_url=competition_url,
        # github_token="your_github_token",  # Optional
        # mcp_api_key="your_mcp_api_key",    # Optional
    )
    
    print("Competition analysis completed!")
    print(f"Detected problem type: {results['competition_info'].problem_type}")
    print(f"Evaluation metric: {results['competition_info'].evaluation_metric}")
    print(f"Analyzed {len(results['github_analysis']['repositories'])} repositories")