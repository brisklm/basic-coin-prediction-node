"""
Project Status Analyzer - Comprehensive Project Understanding System
===================================================================

This module provides complete project status analysis including:
- Codebase structure and architecture mapping
- Dependency analysis and compatibility checking  
- Integration point discovery
- Health assessment and readiness evaluation
- Pre-condition validation for cyclical MCP implementation
"""

import os
import ast
import sys
import json
import yaml
import subprocess
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import pandas as pd
import logging
from collections import defaultdict, Counter
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FileAnalysis:
    """Analysis of a single file"""
    path: str
    type: str  # 'python', 'config', 'data', 'docs', 'other'
    size_bytes: int
    lines: int
    imports: List[str]
    functions: List[str]
    classes: List[str]
    dependencies: List[str]
    complexity_score: float
    last_modified: str
    errors: List[str]

@dataclass
class ModuleAnalysis:
    """Analysis of a Python module"""
    name: str
    file_path: str
    public_api: List[str]
    internal_functions: List[str]
    dependencies: List[str]
    exports: List[str]
    docstring: Optional[str]
    test_coverage: Optional[float]
    integration_points: List[str]

@dataclass
class ProjectStructure:
    """Overall project structure analysis"""
    root_directory: str
    total_files: int
    total_lines: int
    total_size_bytes: int
    file_types: Dict[str, int]
    directory_structure: Dict[str, Any]
    main_modules: List[str]
    entry_points: List[str]
    configuration_files: List[str]

@dataclass
class DependencyAnalysis:
    """Dependency analysis results"""
    direct_dependencies: Dict[str, str]  # package: version
    indirect_dependencies: Dict[str, str]
    missing_dependencies: List[str]
    version_conflicts: List[Dict[str, Any]]
    security_issues: List[Dict[str, Any]]
    compatibility_matrix: Dict[str, Dict[str, bool]]

@dataclass
class IntegrationPoint:
    """Integration point between modules"""
    source_module: str
    target_module: str
    integration_type: str  # 'import', 'inheritance', 'composition', 'function_call'
    details: Dict[str, Any]
    risk_level: str  # 'low', 'medium', 'high'

@dataclass
class ProjectHealth:
    """Overall project health assessment"""
    health_score: float  # 0-100
    readiness_level: str  # 'not_ready', 'basic', 'ready', 'production_ready'
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    mcp_readiness: bool
    cyclical_optimization_compatible: bool

@dataclass
class ProjectStatus:
    """Complete project status"""
    analysis_timestamp: str
    project_structure: ProjectStructure
    file_analyses: List[FileAnalysis]
    module_analyses: List[ModuleAnalysis]
    dependency_analysis: DependencyAnalysis
    integration_points: List[IntegrationPoint]
    project_health: ProjectHealth
    readiness_assessment: Dict[str, Any]

class ProjectStatusAnalyzer:
    """Comprehensive project status analyzer"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.analysis_cache = {}
        self.supported_extensions = {
            '.py': 'python',
            '.yaml': 'config',
            '.yml': 'config',
            '.json': 'config',
            '.toml': 'config',
            '.cfg': 'config',
            '.ini': 'config',
            '.csv': 'data',
            '.txt': 'docs',
            '.md': 'docs',
            '.rst': 'docs'
        }
    
    def analyze_project(self) -> ProjectStatus:
        """Perform comprehensive project analysis"""
        logger.info("🔍 Starting comprehensive project status analysis...")
        
        # Step 1: Analyze project structure
        logger.info("📁 Analyzing project structure...")
        project_structure = self._analyze_project_structure()
        
        # Step 2: Analyze individual files
        logger.info("📄 Analyzing individual files...")
        file_analyses = self._analyze_files()
        
        # Step 3: Analyze Python modules
        logger.info("🐍 Analyzing Python modules...")
        module_analyses = self._analyze_modules()
        
        # Step 4: Analyze dependencies
        logger.info("📦 Analyzing dependencies...")
        dependency_analysis = self._analyze_dependencies()
        
        # Step 5: Discover integration points
        logger.info("🔗 Discovering integration points...")
        integration_points = self._discover_integration_points(module_analyses)
        
        # Step 6: Assess project health
        logger.info("🏥 Assessing project health...")
        project_health = self._assess_project_health(
            project_structure, file_analyses, module_analyses, 
            dependency_analysis, integration_points
        )
        
        # Step 7: Generate readiness assessment
        logger.info("🎯 Generating readiness assessment...")
        readiness_assessment = self._generate_readiness_assessment(
            project_structure, module_analyses, dependency_analysis, project_health
        )
        
        project_status = ProjectStatus(
            analysis_timestamp=pd.Timestamp.now().isoformat(),
            project_structure=project_structure,
            file_analyses=file_analyses,
            module_analyses=module_analyses,
            dependency_analysis=dependency_analysis,
            integration_points=integration_points,
            project_health=project_health,
            readiness_assessment=readiness_assessment
        )
        
        logger.info("✅ Project status analysis completed!")
        return project_status
    
    def _analyze_project_structure(self) -> ProjectStructure:
        """Analyze overall project structure"""
        
        all_files = []
        directory_structure = {}
        file_types = Counter()
        total_size = 0
        total_lines = 0
        
        # Walk through project directory
        for root, dirs, files in os.walk(self.project_root):
            # Skip common ignored directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            rel_root = os.path.relpath(root, self.project_root)
            if rel_root == '.':
                rel_root = ''
            
            for file in files:
                if file.startswith('.'):
                    continue
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.project_root)
                
                # Get file info
                try:
                    stat = os.stat(file_path)
                    file_size = stat.st_size
                    total_size += file_size
                    
                    # Count lines for text files
                    file_ext = Path(file).suffix.lower()
                    if file_ext in self.supported_extensions:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = len(f.readlines())
                                total_lines += lines
                        except:
                            lines = 0
                    
                    # Categorize file type
                    file_type = self.supported_extensions.get(file_ext, 'other')
                    file_types[file_type] += 1
                    
                    all_files.append(rel_path)
                    
                except OSError:
                    continue
        
        # Identify main modules and entry points
        main_modules = []
        entry_points = []
        config_files = []
        
        for file_path in all_files:
            if file_path.endswith('.py'):
                if any(name in file_path.lower() for name in ['main', 'app', 'run', 'start']):
                    entry_points.append(file_path)
                if not '/' in file_path or file_path.count('/') == 1:  # Top level or one level deep
                    main_modules.append(file_path)
            elif any(file_path.endswith(ext) for ext in ['.yaml', '.yml', '.json', '.toml', '.cfg']):
                config_files.append(file_path)
        
        return ProjectStructure(
            root_directory=str(self.project_root),
            total_files=len(all_files),
            total_lines=total_lines,
            total_size_bytes=total_size,
            file_types=dict(file_types),
            directory_structure=directory_structure,
            main_modules=main_modules,
            entry_points=entry_points,
            configuration_files=config_files
        )
    
    def _analyze_files(self) -> List[FileAnalysis]:
        """Analyze individual files"""
        
        file_analyses = []
        
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__']]
            
            for file in files:
                if file.startswith('.'):
                    continue
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.project_root)
                
                try:
                    analysis = self._analyze_single_file(file_path, rel_path)
                    if analysis:
                        file_analyses.append(analysis)
                except Exception as e:
                    logger.warning(f"Failed to analyze file {rel_path}: {e}")
        
        return file_analyses
    
    def _analyze_single_file(self, file_path: str, rel_path: str) -> Optional[FileAnalysis]:
        """Analyze a single file"""
        
        try:
            stat = os.stat(file_path)
            file_size = stat.st_size
            file_ext = Path(file_path).suffix.lower()
            file_type = self.supported_extensions.get(file_ext, 'other')
            
            imports = []
            functions = []
            classes = []
            dependencies = []
            lines = 0
            complexity_score = 0.0
            errors = []
            
            if file_ext == '.py':
                # Analyze Python file
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = len(content.splitlines())
                    
                    # Parse AST
                    try:
                        tree = ast.parse(content)
                        imports, functions, classes, dependencies, complexity_score = self._analyze_python_ast(tree)
                    except SyntaxError as e:
                        errors.append(f"Syntax error: {e}")
                        
                except Exception as e:
                    errors.append(f"Read error: {e}")
            
            elif file_type in ['config', 'docs', 'data']:
                # Count lines for other text files
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                except:
                    lines = 0
            
            return FileAnalysis(
                path=rel_path,
                type=file_type,
                size_bytes=file_size,
                lines=lines,
                imports=imports,
                functions=functions,
                classes=classes,
                dependencies=dependencies,
                complexity_score=complexity_score,
                last_modified=pd.Timestamp.fromtimestamp(stat.st_mtime).isoformat(),
                errors=errors
            )
            
        except Exception as e:
            logger.warning(f"Error analyzing file {rel_path}: {e}")
            return None
    
    def _analyze_python_ast(self, tree: ast.AST) -> Tuple[List[str], List[str], List[str], List[str], float]:
        """Analyze Python AST for imports, functions, classes, etc."""
        
        imports = []
        functions = []
        classes = []
        dependencies = set()
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
                    dependencies.add(alias.name.split('.')[0])
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(f"from {node.module}")
                    dependencies.add(node.module.split('.')[0])
                for alias in node.names:
                    imports.append(alias.name)
            
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
                complexity += len(node.body)
            
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
                complexity += len(node.body)
            
            elif isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
        
        # Normalize complexity score
        complexity_score = min(10.0, complexity / 10.0)
        
        return imports, functions, classes, list(dependencies), complexity_score
    
    def _analyze_modules(self) -> List[ModuleAnalysis]:
        """Analyze Python modules in detail"""
        
        module_analyses = []
        
        # Find all Python files
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__']]
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.project_root)
                    python_files.append((file_path, rel_path))
        
        for file_path, rel_path in python_files:
            try:
                analysis = self._analyze_python_module(file_path, rel_path)
                if analysis:
                    module_analyses.append(analysis)
            except Exception as e:
                logger.warning(f"Failed to analyze module {rel_path}: {e}")
        
        return module_analyses
    
    def _analyze_python_module(self, file_path: str, rel_path: str) -> Optional[ModuleAnalysis]:
        """Analyze a single Python module"""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract module info
            docstring = ast.get_docstring(tree)
            
            public_api = []
            internal_functions = []
            dependencies = set()
            exports = []
            integration_points = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith('_'):
                        internal_functions.append(node.name)
                    else:
                        public_api.append(node.name)
                
                elif isinstance(node, ast.ClassDef):
                    if node.name.startswith('_'):
                        internal_functions.append(node.name)
                    else:
                        public_api.append(node.name)
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.add(alias.name.split('.')[0])
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.add(node.module.split('.')[0])
            
            # Look for __all__ exports
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == '__all__':
                            if isinstance(node.value, ast.List):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Str):
                                        exports.append(elt.s)
                                    elif isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        exports.append(elt.value)
            
            # Identify integration points
            for dep in dependencies:
                if any(dep in mod_file for mod_file, _ in [(file_path, rel_path)]):
                    integration_points.append(dep)
            
            module_name = rel_path.replace('/', '.').replace('.py', '')
            
            return ModuleAnalysis(
                name=module_name,
                file_path=rel_path,
                public_api=public_api,
                internal_functions=internal_functions,
                dependencies=list(dependencies),
                exports=exports,
                docstring=docstring,
                test_coverage=None,  # Would need coverage tool integration
                integration_points=integration_points
            )
            
        except Exception as e:
            logger.warning(f"Error analyzing module {rel_path}: {e}")
            return None
    
    def _analyze_dependencies(self) -> DependencyAnalysis:
        """Analyze project dependencies"""
        
        # Read requirements files
        direct_deps = {}
        req_files = ['requirements.txt', 'pyproject.toml', 'setup.py', 'Pipfile']
        
        for req_file in req_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                try:
                    if req_file == 'requirements.txt':
                        direct_deps.update(self._parse_requirements_txt(req_path))
                    elif req_file == 'pyproject.toml':
                        direct_deps.update(self._parse_pyproject_toml(req_path))
                except Exception as e:
                    logger.warning(f"Error parsing {req_file}: {e}")
        
        # Get installed packages
        installed_packages = self._get_installed_packages()
        
        # Find missing dependencies
        missing_deps = []
        # Create case-insensitive lookup for installed packages
        installed_lower = {pkg.lower(): pkg for pkg in installed_packages.keys()}
        
        for dep in direct_deps:
            if dep not in installed_packages and dep.lower() not in installed_lower:
                missing_deps.append(dep)
        
        # Check for version conflicts
        version_conflicts = []
        for dep, required_version in direct_deps.items():
            if dep in installed_packages:
                installed_version = installed_packages[dep]
                if required_version and installed_version != required_version:
                    version_conflicts.append({
                        'package': dep,
                        'required': required_version,
                        'installed': installed_version
                    })
        
        # Basic security check (would need vulnerability database for real implementation)
        security_issues = []
        
        # Compatibility matrix (simplified)
        compatibility_matrix = {}
        for dep in direct_deps:
            compatibility_matrix[dep] = {
                'python_3_8': True,
                'python_3_9': True,
                'python_3_10': True,
                'python_3_11': True
            }
        
        return DependencyAnalysis(
            direct_dependencies=direct_deps,
            indirect_dependencies=installed_packages,
            missing_dependencies=missing_deps,
            version_conflicts=version_conflicts,
            security_issues=security_issues,
            compatibility_matrix=compatibility_matrix
        )
    
    def _parse_requirements_txt(self, file_path: Path) -> Dict[str, str]:
        """Parse requirements.txt file"""
        deps = {}
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '>=' in line:
                            name, version = line.split('>=')
                            deps[name.strip()] = version.strip()
                        elif '==' in line:
                            name, version = line.split('==')
                            deps[name.strip()] = version.strip()
                        else:
                            deps[line] = ''
        except Exception as e:
            logger.warning(f"Error parsing requirements.txt: {e}")
        
        return deps
    
    def _parse_pyproject_toml(self, file_path: Path) -> Dict[str, str]:
        """Parse pyproject.toml file"""
        deps = {}
        try:
            import toml
            with open(file_path, 'r') as f:
                data = toml.load(f)
            
            # Look for dependencies in different sections
            if 'project' in data and 'dependencies' in data['project']:
                for dep in data['project']['dependencies']:
                    if '>=' in dep:
                        name, version = dep.split('>=')
                        deps[name.strip()] = version.strip()
                    elif '==' in dep:
                        name, version = dep.split('==')
                        deps[name.strip()] = version.strip()
                    else:
                        deps[dep] = ''
        except ImportError:
            logger.warning("toml package not available for parsing pyproject.toml")
        except Exception as e:
            logger.warning(f"Error parsing pyproject.toml: {e}")
        
        return deps
    
    def _get_installed_packages(self) -> Dict[str, str]:
        """Get currently installed packages"""
        installed = {}
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[2:]  # Skip header
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        installed[parts[0]] = parts[1]
        except Exception as e:
            logger.warning(f"Error getting installed packages: {e}")
        
        return installed
    
    def _discover_integration_points(self, module_analyses: List[ModuleAnalysis]) -> List[IntegrationPoint]:
        """Discover integration points between modules"""
        
        integration_points = []
        
        # Create module lookup
        modules_by_name = {mod.name: mod for mod in module_analyses}
        
        for module in module_analyses:
            for dep in module.dependencies:
                # Check if dependency is internal module
                for other_module in module_analyses:
                    if (dep in other_module.name or 
                        any(dep in api_item for api_item in other_module.public_api)):
                        
                        integration_point = IntegrationPoint(
                            source_module=module.name,
                            target_module=other_module.name,
                            integration_type='import',
                            details={'dependency': dep},
                            risk_level=self._assess_integration_risk(module, other_module)
                        )
                        integration_points.append(integration_point)
        
        return integration_points
    
    def _assess_integration_risk(self, source_module: ModuleAnalysis, 
                               target_module: ModuleAnalysis) -> str:
        """Assess risk level of integration between modules"""
        
        # Simple heuristic based on module complexity
        source_complexity = len(source_module.public_api) + len(source_module.dependencies)
        target_complexity = len(target_module.public_api) + len(target_module.dependencies)
        
        total_complexity = source_complexity + target_complexity
        
        if total_complexity > 20:
            return 'high'
        elif total_complexity > 10:
            return 'medium'
        else:
            return 'low'
    
    def _assess_project_health(self, structure: ProjectStructure, 
                             files: List[FileAnalysis],
                             modules: List[ModuleAnalysis],
                             dependencies: DependencyAnalysis,
                             integrations: List[IntegrationPoint]) -> ProjectHealth:
        """Assess overall project health"""
        
        health_score = 100.0
        critical_issues = []
        warnings = []
        recommendations = []
        
        # Check for critical issues
        if dependencies.missing_dependencies:
            critical_issues.append(f"Missing dependencies: {dependencies.missing_dependencies}")
            health_score -= 20
        
        if dependencies.version_conflicts:
            critical_issues.append(f"Version conflicts: {len(dependencies.version_conflicts)} packages")
            health_score -= 15
        
        # Check for errors in files
        error_files = [f for f in files if f.errors]
        if error_files:
            critical_issues.append(f"Files with errors: {len(error_files)}")
            health_score -= 10
        
        # Check for high complexity
        complex_files = [f for f in files if f.complexity_score > 7.0]
        if complex_files:
            warnings.append(f"High complexity files: {len(complex_files)}")
            health_score -= 5
        
        # Check for high-risk integrations
        high_risk_integrations = [i for i in integrations if i.risk_level == 'high']
        if high_risk_integrations:
            warnings.append(f"High-risk integrations: {len(high_risk_integrations)}")
            health_score -= 5
        
        # Generate recommendations
        if not structure.entry_points:
            recommendations.append("Consider adding clear entry points (main.py, app.py)")
        
        if structure.file_types.get('docs', 0) == 0:
            recommendations.append("Consider adding documentation files")
        
        if len(modules) > 10 and not any('test' in m.name for m in modules):
            recommendations.append("Consider adding unit tests")
        
        # Determine readiness level
        if health_score >= 90:
            readiness_level = 'production_ready'
        elif health_score >= 75:
            readiness_level = 'ready'
        elif health_score >= 60:
            readiness_level = 'basic'
        else:
            readiness_level = 'not_ready'
        
        # Assess MCP readiness
        mcp_ready = (
            health_score >= 70 and
            not critical_issues and
            len(dependencies.missing_dependencies) == 0
        )
        
        # Assess cyclical optimization compatibility
        cyclical_compatible = (
            mcp_ready and
            any('enhanced_competition_toolkit' in m.name for m in modules) and
            any('competition_analyzer' in m.name for m in modules)
        )
        
        return ProjectHealth(
            health_score=max(0, health_score),
            readiness_level=readiness_level,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations,
            mcp_readiness=mcp_ready,
            cyclical_optimization_compatible=cyclical_compatible
        )
    
    def _generate_readiness_assessment(self, structure: ProjectStructure,
                                     modules: List[ModuleAnalysis],
                                     dependencies: DependencyAnalysis,
                                     health: ProjectHealth) -> Dict[str, Any]:
        """Generate comprehensive readiness assessment"""
        
        # Core toolkit components check
        core_components = {
            'ai_competition_toolkit': any('ai_competition_toolkit' in m.name for m in modules),
            'enhanced_competition_toolkit': any('enhanced_competition_toolkit' in m.name for m in modules),
            'competition_analyzer': any('competition_analyzer' in m.name for m in modules),
            'cyclical_mcp_system': any('cyclical_mcp_system' in m.name for m in modules)
        }
        
        # Required dependencies check
        required_deps = [
            'pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm', 
            'optuna', 'requests', 'beautifulsoup4'
        ]
        
        deps_status = {}
        for dep in required_deps:
            deps_status[dep] = dep in dependencies.direct_dependencies
        
        # MCP server prerequisites
        mcp_prerequisites = {
            'anthropic_support': 'anthropic' in dependencies.direct_dependencies,
            'openai_support': 'openai' in dependencies.direct_dependencies,
            'async_support': any('asyncio' in str(m.dependencies) for m in modules),
            'config_management': len(structure.configuration_files) > 0
        }
        
        # Cyclical optimization readiness
        cyclical_readiness = {
            'core_framework': core_components['ai_competition_toolkit'],
            'enhanced_framework': core_components['enhanced_competition_toolkit'],
            'competition_analyzer': core_components['competition_analyzer'],
            'cyclical_system': core_components['cyclical_mcp_system'],
            'dependency_satisfaction': len(dependencies.missing_dependencies) == 0,
            'integration_health': len([i for i in health.critical_issues if 'integration' in i.lower()]) == 0
        }
        
        # Overall readiness score
        component_score = sum(core_components.values()) / len(core_components) * 100
        dependency_score = sum(deps_status.values()) / len(deps_status) * 100
        mcp_score = sum(mcp_prerequisites.values()) / len(mcp_prerequisites) * 100
        cyclical_score = sum(cyclical_readiness.values()) / len(cyclical_readiness) * 100
        
        overall_readiness = (component_score + dependency_score + mcp_score + cyclical_score) / 4
        
        return {
            'overall_readiness_score': overall_readiness,
            'component_readiness': core_components,
            'dependency_status': deps_status,
            'mcp_prerequisites': mcp_prerequisites,
            'cyclical_optimization_readiness': cyclical_readiness,
            'readiness_breakdown': {
                'components': component_score,
                'dependencies': dependency_score,
                'mcp_support': mcp_score,
                'cyclical_support': cyclical_score
            },
            'next_steps': self._generate_next_steps(
                core_components, deps_status, mcp_prerequisites, cyclical_readiness
            )
        }
    
    def _generate_next_steps(self, components: Dict[str, bool], 
                           dependencies: Dict[str, bool],
                           mcp_prereqs: Dict[str, bool],
                           cyclical_readiness: Dict[str, bool]) -> List[str]:
        """Generate actionable next steps"""
        
        next_steps = []
        
        # Component-related steps
        missing_components = [name for name, present in components.items() if not present]
        if missing_components:
            next_steps.append(f"Implement missing components: {', '.join(missing_components)}")
        
        # Dependency-related steps
        missing_deps = [name for name, present in dependencies.items() if not present]
        if missing_deps:
            next_steps.append(f"Install missing dependencies: pip install {' '.join(missing_deps)}")
        
        # MCP-related steps
        if not mcp_prereqs['anthropic_support'] and not mcp_prereqs['openai_support']:
            next_steps.append("Install AI service dependencies: pip install anthropic openai")
        
        if not mcp_prereqs['async_support']:
            next_steps.append("Add async/await support to enable cyclical optimization")
        
        # Cyclical optimization steps
        if not cyclical_readiness['cyclical_system']:
            next_steps.append("Implement cyclical MCP system (cyclical_mcp_system.py)")
        
        if not cyclical_readiness['integration_health']:
            next_steps.append("Resolve integration issues between components")
        
        return next_steps
    
    def save_analysis(self, status: ProjectStatus, output_file: str = "project_status_report.json"):
        """Save analysis results to file"""
        
        # Convert to serializable format
        status_dict = asdict(status)
        
        with open(output_file, 'w') as f:
            json.dump(status_dict, f, indent=2, default=str)
        
        logger.info(f"📊 Project status report saved to {output_file}")
    
    def generate_report(self, status: ProjectStatus) -> str:
        """Generate human-readable status report"""
        
        report = []
        report.append("🔍 PROJECT STATUS ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Project overview
        report.append("📁 PROJECT OVERVIEW")
        report.append("-" * 20)
        report.append(f"Root Directory: {status.project_structure.root_directory}")
        report.append(f"Total Files: {status.project_structure.total_files:,}")
        report.append(f"Total Lines: {status.project_structure.total_lines:,}")
        report.append(f"Total Size: {status.project_structure.total_size_bytes / 1024 / 1024:.1f} MB")
        report.append("")
        
        # File types
        report.append("📄 FILE TYPES")
        report.append("-" * 15)
        for file_type, count in status.project_structure.file_types.items():
            report.append(f"  {file_type}: {count}")
        report.append("")
        
        # Health assessment
        health = status.project_health
        report.append("🏥 PROJECT HEALTH")
        report.append("-" * 17)
        report.append(f"Health Score: {health.health_score:.1f}/100")
        report.append(f"Readiness Level: {health.readiness_level}")
        report.append(f"MCP Ready: {'✅' if health.mcp_readiness else '❌'}")
        report.append(f"Cyclical Optimization Compatible: {'✅' if health.cyclical_optimization_compatible else '❌'}")
        report.append("")
        
        # Critical issues
        if health.critical_issues:
            report.append("🚨 CRITICAL ISSUES")
            report.append("-" * 17)
            for issue in health.critical_issues:
                report.append(f"  ❌ {issue}")
            report.append("")
        
        # Warnings
        if health.warnings:
            report.append("⚠️ WARNINGS")
            report.append("-" * 12)
            for warning in health.warnings:
                report.append(f"  ⚠️ {warning}")
            report.append("")
        
        # Dependencies
        deps = status.dependency_analysis
        report.append("📦 DEPENDENCIES")
        report.append("-" * 15)
        report.append(f"Direct Dependencies: {len(deps.direct_dependencies)}")
        report.append(f"Missing Dependencies: {len(deps.missing_dependencies)}")
        report.append(f"Version Conflicts: {len(deps.version_conflicts)}")
        
        if deps.missing_dependencies:
            report.append("Missing:")
            for dep in deps.missing_dependencies:
                report.append(f"  - {dep}")
        report.append("")
        
        # Readiness assessment
        readiness = status.readiness_assessment
        report.append("🎯 READINESS ASSESSMENT")
        report.append("-" * 23)
        report.append(f"Overall Readiness: {readiness['overall_readiness_score']:.1f}%")
        report.append("")
        
        report.append("Component Status:")
        for comp, status_val in readiness['component_readiness'].items():
            status_icon = "✅" if status_val else "❌"
            report.append(f"  {status_icon} {comp}")
        report.append("")
        
        # Next steps
        if readiness['next_steps']:
            report.append("🚀 NEXT STEPS")
            report.append("-" * 12)
            for i, step in enumerate(readiness['next_steps'], 1):
                report.append(f"  {i}. {step}")
            report.append("")
        
        return "\n".join(report)

def analyze_project_status(project_root: str = ".") -> ProjectStatus:
    """Convenience function to analyze project status"""
    analyzer = ProjectStatusAnalyzer(project_root)
    return analyzer.analyze_project()

def generate_status_report(project_root: str = ".", 
                         save_json: bool = True,
                         print_report: bool = True) -> ProjectStatus:
    """Generate comprehensive project status report"""
    
    logger.info("🔍 Starting comprehensive project status analysis...")
    
    # Analyze project
    status = analyze_project_status(project_root)
    
    # Generate report
    analyzer = ProjectStatusAnalyzer(project_root)
    report = analyzer.generate_report(status)
    
    if print_report:
        print(report)
    
    if save_json:
        analyzer.save_analysis(status)
    
    # Generate recommendations
    health = status.project_health
    if health.cyclical_optimization_compatible:
        logger.info("✅ Project is ready for cyclical MCP optimization!")
    elif health.mcp_readiness:
        logger.info("⚠️ Project is MCP-ready but needs cyclical optimization setup")
    else:
        logger.info("❌ Project needs significant work before MCP implementation")
    
    return status

if __name__ == "__main__":
    # Run comprehensive project analysis
    status = generate_status_report()
    
    print("\n🎯 SUMMARY:")
    print(f"Health Score: {status.project_health.health_score:.1f}/100")
    print(f"MCP Ready: {status.project_health.mcp_readiness}")
    print(f"Cyclical Optimization Ready: {status.project_health.cyclical_optimization_compatible}")
    print(f"Overall Readiness: {status.readiness_assessment['overall_readiness_score']:.1f}%")