"""
Pre-condition Validator for Cyclical MCP Implementation
======================================================

This module validates that all preconditions are met before implementing
the cyclical MCP optimization system. It ensures proper project structure,
dependencies, integration points, and system readiness.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import importlib
import sys
import json

from project_status_analyzer import ProjectStatusAnalyzer, ProjectStatus, generate_status_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PreConditionResult:
    """Result of a single pre-condition check"""
    name: str
    passed: bool
    description: str
    details: Dict[str, Any]
    severity: str  # 'critical', 'warning', 'info'
    fix_suggestions: List[str]

@dataclass
class ValidationReport:
    """Complete validation report"""
    overall_status: str  # 'ready', 'needs_fixes', 'not_ready'
    total_checks: int
    passed_checks: int
    critical_failures: int
    warnings: int
    results: List[PreConditionResult]
    ready_for_mcp: bool
    ready_for_cyclical: bool
    next_steps: List[str]

class PreConditionValidator:
    """Validates all pre-conditions for cyclical MCP implementation"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.status_analyzer = ProjectStatusAnalyzer(project_root)
        
    async def validate_all_preconditions(self) -> ValidationReport:
        """Validate all pre-conditions for cyclical MCP implementation"""
        
        logger.info("🔍 Starting comprehensive pre-condition validation...")
        
        # Get project status
        project_status = self.status_analyzer.analyze_project()
        
        # Define all validation checks
        validation_checks = [
            self._check_core_components,
            self._check_dependencies,
            self._check_integration_points,
            self._check_python_environment,
            self._check_async_support,
            self._check_mcp_prerequisites,
            self._check_competition_framework,
            self._check_file_structure,
            self._check_configuration_management,
            self._check_error_handling
        ]
        
        results = []
        
        # Run all validation checks
        for check in validation_checks:
            try:
                result = await check(project_status)
                results.append(result)
                
                status_icon = "✅" if result.passed else "❌"
                logger.info(f"{status_icon} {result.name}")
                
            except Exception as e:
                error_result = PreConditionResult(
                    name=check.__name__,
                    passed=False,
                    description=f"Validation check failed: {e}",
                    details={"error": str(e)},
                    severity="critical",
                    fix_suggestions=[f"Fix validation error in {check.__name__}"]
                )
                results.append(error_result)
                logger.error(f"❌ {check.__name__} failed: {e}")
        
        # Generate summary
        total_checks = len(results)
        passed_checks = sum(1 for r in results if r.passed)
        critical_failures = sum(1 for r in results if not r.passed and r.severity == 'critical')
        warnings = sum(1 for r in results if not r.passed and r.severity == 'warning')
        
        # Determine overall status
        if critical_failures == 0 and warnings == 0:
            overall_status = 'ready'
        elif critical_failures == 0:
            overall_status = 'needs_fixes'
        else:
            overall_status = 'not_ready'
        
        # Determine readiness levels
        ready_for_mcp = critical_failures == 0 and passed_checks >= total_checks * 0.8
        ready_for_cyclical = ready_for_mcp and passed_checks >= total_checks * 0.9
        
        # Generate next steps
        next_steps = self._generate_next_steps(results, project_status)
        
        validation_report = ValidationReport(
            overall_status=overall_status,
            total_checks=total_checks,
            passed_checks=passed_checks,
            critical_failures=critical_failures,
            warnings=warnings,
            results=results,
            ready_for_mcp=ready_for_mcp,
            ready_for_cyclical=ready_for_cyclical,
            next_steps=next_steps
        )
        
        logger.info("✅ Pre-condition validation completed!")
        return validation_report
    
    async def _check_core_components(self, status: ProjectStatus) -> PreConditionResult:
        """Check that all core components are present"""
        
        required_components = {
            'ai_competition_toolkit': 'Core competition framework',
            'enhanced_competition_toolkit': 'Enhanced framework with autonomous features',
            'competition_analyzer': 'Competition analysis and GitHub learning',
            'cyclical_mcp_system': 'Cyclical MCP optimization system',
            'project_status_analyzer': 'Project status analysis'
        }
        
        missing_components = []
        present_components = []
        
        for module in status.module_analyses:
            for component in required_components:
                if component in module.name:
                    present_components.append(component)
                    break
        
        for component in required_components:
            if component not in present_components:
                missing_components.append(component)
        
        passed = len(missing_components) == 0
        severity = 'critical' if missing_components else 'info'
        
        fix_suggestions = []
        if missing_components:
            fix_suggestions = [f"Implement missing component: {comp}" for comp in missing_components]
        
        return PreConditionResult(
            name="Core Components Check",
            passed=passed,
            description=f"Validates presence of all required core components",
            details={
                "required": list(required_components.keys()),
                "present": present_components,
                "missing": missing_components
            },
            severity=severity,
            fix_suggestions=fix_suggestions
        )
    
    async def _check_dependencies(self, status: ProjectStatus) -> PreConditionResult:
        """Check that all required dependencies are available"""
        
        critical_deps = [
            'numpy', 'pandas', 'sklearn', 'optuna', 'requests', 'yaml'
        ]
        
        optional_deps = [
            'xgboost', 'lightgbm', 'catboost', 'matplotlib', 'plotly', 
            'beautifulsoup4', 'joblib', 'tqdm'
        ]
        
        missing_critical = []
        missing_optional = []
        
        # Check critical dependencies with correct module names
        module_map = {
            'scikit-learn': 'sklearn',
            'pyyaml': 'yaml',
            'beautifulsoup4': 'bs4'
        }
        
        for dep in critical_deps:
            module_name = module_map.get(dep, dep.replace('-', '_'))
            try:
                importlib.import_module(module_name)
            except ImportError:
                missing_critical.append(dep)
        
        # Check optional dependencies
        for dep in optional_deps:
            module_name = module_map.get(dep, dep.replace('-', '_'))
            try:
                importlib.import_module(module_name)
            except ImportError:
                missing_optional.append(dep)
        
        passed = len(missing_critical) == 0
        severity = 'critical' if missing_critical else 'warning' if missing_optional else 'info'
        
        fix_suggestions = []
        if missing_critical:
            fix_suggestions.append(f"Install critical dependencies: pip install {' '.join(missing_critical)}")
        if missing_optional:
            fix_suggestions.append(f"Install optional dependencies: pip install {' '.join(missing_optional)}")
        
        return PreConditionResult(
            name="Dependencies Check",
            passed=passed,
            description="Validates all required dependencies are installed",
            details={
                "critical_missing": missing_critical,
                "optional_missing": missing_optional,
                "version_conflicts": status.dependency_analysis.version_conflicts
            },
            severity=severity,
            fix_suggestions=fix_suggestions
        )
    
    async def _check_integration_points(self, status: ProjectStatus) -> PreConditionResult:
        """Check integration points between modules"""
        
        high_risk_integrations = [
            i for i in status.integration_points if i.risk_level == 'high'
        ]
        
        critical_integrations = [
            'ai_competition_toolkit -> enhanced_competition_toolkit',
            'enhanced_competition_toolkit -> competition_analyzer',
            'enhanced_competition_toolkit -> cyclical_mcp_system'
        ]
        
        found_integrations = []
        for integration in status.integration_points:
            integration_desc = f"{integration.source_module} -> {integration.target_module}"
            for critical in critical_integrations:
                if any(comp in integration_desc for comp in critical.split(' -> ')):
                    found_integrations.append(integration_desc)
        
        passed = len(high_risk_integrations) == 0
        severity = 'warning' if high_risk_integrations else 'info'
        
        fix_suggestions = []
        if high_risk_integrations:
            fix_suggestions.append("Review and simplify high-risk integrations")
            fix_suggestions.append("Consider refactoring complex module dependencies")
        
        return PreConditionResult(
            name="Integration Points Check",
            passed=passed,
            description="Validates module integration points are healthy",
            details={
                "high_risk_count": len(high_risk_integrations),
                "critical_integrations_found": found_integrations,
                "total_integrations": len(status.integration_points)
            },
            severity=severity,
            fix_suggestions=fix_suggestions
        )
    
    async def _check_python_environment(self, status: ProjectStatus) -> PreConditionResult:
        """Check Python environment compatibility"""
        
        python_version = sys.version_info
        min_version = (3, 8)
        recommended_version = (3, 9)
        
        version_compatible = python_version >= min_version
        version_recommended = python_version >= recommended_version
        
        passed = version_compatible
        severity = 'critical' if not version_compatible else 'warning' if not version_recommended else 'info'
        
        fix_suggestions = []
        if not version_compatible:
            fix_suggestions.append(f"Upgrade Python to version {min_version[0]}.{min_version[1]} or higher")
        elif not version_recommended:
            fix_suggestions.append(f"Consider upgrading to Python {recommended_version[0]}.{recommended_version[1]} or higher for better performance")
        
        return PreConditionResult(
            name="Python Environment Check",
            passed=passed,
            description="Validates Python version compatibility",
            details={
                "current_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                "minimum_required": f"{min_version[0]}.{min_version[1]}",
                "recommended": f"{recommended_version[0]}.{recommended_version[1]}",
                "compatible": version_compatible
            },
            severity=severity,
            fix_suggestions=fix_suggestions
        )
    
    async def _check_async_support(self, status: ProjectStatus) -> PreConditionResult:
        """Check async/await support for cyclical optimization"""
        
        async_imports = []
        async_functions = []
        
        for module in status.module_analyses:
            if 'asyncio' in module.dependencies:
                async_imports.append(module.name)
            
            # Check for async functions in public API
            for func in module.public_api:
                if 'async' in func or func.startswith('a'):  # Simple heuristic
                    async_functions.append(f"{module.name}.{func}")
        
        has_async_support = len(async_imports) > 0
        
        passed = has_async_support
        severity = 'warning' if not has_async_support else 'info'
        
        fix_suggestions = []
        if not has_async_support:
            fix_suggestions.append("Add asyncio support to enable cyclical optimization")
            fix_suggestions.append("Convert key functions to async/await pattern")
        
        return PreConditionResult(
            name="Async Support Check",
            passed=passed,
            description="Validates async/await support for cyclical operations",
            details={
                "modules_with_async": async_imports,
                "async_functions": async_functions,
                "async_capable": has_async_support
            },
            severity=severity,
            fix_suggestions=fix_suggestions
        )
    
    async def _check_mcp_prerequisites(self, status: ProjectStatus) -> PreConditionResult:
        """Check MCP (Model Context Protocol) prerequisites"""
        
        mcp_deps = ['anthropic', 'openai']
        json_support = True
        config_support = len(status.project_structure.configuration_files) > 0
        
        available_mcp_deps = []
        missing_mcp_deps = []
        
        for dep in mcp_deps:
            try:
                importlib.import_module(dep)
                available_mcp_deps.append(dep)
            except ImportError:
                missing_mcp_deps.append(dep)
        
        # Check JSON support
        try:
            import json
        except ImportError:
            json_support = False
        
        has_mcp_support = len(available_mcp_deps) > 0
        
        passed = has_mcp_support and json_support
        severity = 'warning' if not has_mcp_support else 'info'
        
        fix_suggestions = []
        if not has_mcp_support:
            fix_suggestions.append(f"Install MCP dependencies: pip install {' '.join(missing_mcp_deps)}")
        if not config_support:
            fix_suggestions.append("Add configuration file support (YAML/JSON)")
        
        return PreConditionResult(
            name="MCP Prerequisites Check",
            passed=passed,
            description="Validates Model Context Protocol prerequisites",
            details={
                "available_ai_services": available_mcp_deps,
                "missing_ai_services": missing_mcp_deps,
                "json_support": json_support,
                "config_support": config_support
            },
            severity=severity,
            fix_suggestions=fix_suggestions
        )
    
    async def _check_competition_framework(self, status: ProjectStatus) -> PreConditionResult:
        """Check competition framework functionality"""
        
        framework_components = {
            'data_preprocessing': False,
            'feature_engineering': False,
            'model_training': False,
            'ensemble_methods': False,
            'evaluation_metrics': False
        }
        
        # Check for framework components in modules
        for module in status.module_analyses:
            module_content = ' '.join(module.public_api + module.internal_functions).lower()
            
            if any(term in module_content for term in ['preprocess', 'clean', 'transform']):
                framework_components['data_preprocessing'] = True
            
            if any(term in module_content for term in ['feature', 'engineer', 'select']):
                framework_components['feature_engineering'] = True
            
            if any(term in module_content for term in ['train', 'fit', 'model']):
                framework_components['model_training'] = True
            
            if any(term in module_content for term in ['ensemble', 'voting', 'stacking']):
                framework_components['ensemble_methods'] = True
            
            if any(term in module_content for term in ['evaluate', 'metric', 'score']):
                framework_components['evaluation_metrics'] = True
        
        present_components = sum(framework_components.values())
        total_components = len(framework_components)
        
        passed = present_components >= total_components * 0.8
        severity = 'critical' if present_components < total_components * 0.6 else 'warning' if not passed else 'info'
        
        missing_components = [name for name, present in framework_components.items() if not present]
        
        fix_suggestions = []
        if missing_components:
            fix_suggestions = [f"Implement missing framework component: {comp}" for comp in missing_components]
        
        return PreConditionResult(
            name="Competition Framework Check",
            passed=passed,
            description="Validates competition framework completeness",
            details={
                "components": framework_components,
                "completion_rate": f"{present_components}/{total_components}",
                "missing": missing_components
            },
            severity=severity,
            fix_suggestions=fix_suggestions
        )
    
    async def _check_file_structure(self, status: ProjectStatus) -> PreConditionResult:
        """Check project file structure"""
        
        required_files = [
            'requirements.txt',
            '*.py'  # At least some Python files
        ]
        
        recommended_files = [
            'README.md',
            'config.yaml'
        ]
        
        all_files = [f.path for f in status.file_analyses]
        
        missing_required = []
        missing_recommended = []
        
        # Check required files
        has_requirements = any('requirements' in f for f in all_files)
        has_python = any(f.endswith('.py') for f in all_files)
        
        if not has_requirements:
            missing_required.append('requirements.txt')
        if not has_python:
            missing_required.append('Python files')
        
        # Check recommended files
        has_readme = any('readme' in f.lower() for f in all_files)
        has_config = any('config' in f.lower() for f in all_files)
        
        if not has_readme:
            missing_recommended.append('README.md')
        if not has_config:
            missing_recommended.append('config file')
        
        passed = len(missing_required) == 0
        severity = 'critical' if missing_required else 'warning' if missing_recommended else 'info'
        
        fix_suggestions = []
        if missing_required:
            fix_suggestions.extend([f"Create required file: {f}" for f in missing_required])
        if missing_recommended:
            fix_suggestions.extend([f"Consider adding: {f}" for f in missing_recommended])
        
        return PreConditionResult(
            name="File Structure Check",
            passed=passed,
            description="Validates project file structure",
            details={
                "total_files": len(all_files),
                "missing_required": missing_required,
                "missing_recommended": missing_recommended,
                "file_types": status.project_structure.file_types
            },
            severity=severity,
            fix_suggestions=fix_suggestions
        )
    
    async def _check_configuration_management(self, status: ProjectStatus) -> PreConditionResult:
        """Check configuration management capabilities"""
        
        config_files = status.project_structure.configuration_files
        has_yaml_support = False
        has_json_support = False
        
        # Check for YAML/JSON support in modules
        for module in status.module_analyses:
            if 'yaml' in module.dependencies or 'pyyaml' in module.dependencies:
                has_yaml_support = True
            if 'json' in module.dependencies:
                has_json_support = True
        
        # Built-in JSON support
        try:
            import json
            has_json_support = True
        except ImportError:
            pass
        
        has_config_files = len(config_files) > 0
        has_config_support = has_yaml_support or has_json_support
        
        passed = has_config_files and has_config_support
        severity = 'warning' if not passed else 'info'
        
        fix_suggestions = []
        if not has_config_files:
            fix_suggestions.append("Add configuration files (config.yaml, config.json)")
        if not has_config_support:
            fix_suggestions.append("Add configuration parsing support (PyYAML)")
        
        return PreConditionResult(
            name="Configuration Management Check",
            passed=passed,
            description="Validates configuration management capabilities",
            details={
                "config_files": config_files,
                "yaml_support": has_yaml_support,
                "json_support": has_json_support,
                "config_capable": has_config_support
            },
            severity=severity,
            fix_suggestions=fix_suggestions
        )
    
    async def _check_error_handling(self, status: ProjectStatus) -> PreConditionResult:
        """Check error handling and robustness"""
        
        files_with_errors = [f for f in status.file_analyses if f.errors]
        error_count = len(files_with_errors)
        
        # Check for try/except patterns in modules
        modules_with_error_handling = []
        for module in status.module_analyses:
            module_functions = ' '.join(module.public_api + module.internal_functions).lower()
            if any(term in module_functions for term in ['try', 'except', 'error', 'handle']):
                modules_with_error_handling.append(module.name)
        
        passed = error_count == 0
        severity = 'critical' if error_count > 3 else 'warning' if error_count > 0 else 'info'
        
        fix_suggestions = []
        if error_count > 0:
            fix_suggestions.append(f"Fix syntax errors in {error_count} files")
            fix_suggestions.extend([f"Fix errors in {f.path}" for f in files_with_errors[:3]])
        
        return PreConditionResult(
            name="Error Handling Check",
            passed=passed,
            description="Validates code quality and error handling",
            details={
                "files_with_errors": error_count,
                "error_files": [f.path for f in files_with_errors],
                "modules_with_error_handling": modules_with_error_handling
            },
            severity=severity,
            fix_suggestions=fix_suggestions
        )
    
    def _generate_next_steps(self, results: List[PreConditionResult], 
                           status: ProjectStatus) -> List[str]:
        """Generate actionable next steps based on validation results"""
        
        next_steps = []
        
        # Priority 1: Critical failures
        critical_results = [r for r in results if not r.passed and r.severity == 'critical']
        for result in critical_results:
            next_steps.extend(result.fix_suggestions)
        
        # Priority 2: Warnings that affect MCP readiness
        warning_results = [r for r in results if not r.passed and r.severity == 'warning']
        for result in warning_results[:3]:  # Limit to top 3 warnings
            next_steps.extend(result.fix_suggestions[:2])  # Top 2 suggestions per warning
        
        # Priority 3: General improvements
        if len(critical_results) == 0 and len(warning_results) == 0:
            next_steps.append("All pre-conditions met! Ready for cyclical MCP implementation")
            next_steps.append("Consider running cyclical optimization tests")
            next_steps.append("Set up monitoring and logging for production use")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_steps = []
        for step in next_steps:
            if step not in seen:
                seen.add(step)
                unique_steps.append(step)
        
        return unique_steps[:10]  # Limit to top 10 actionable steps
    
    def generate_report(self, validation_report: ValidationReport) -> str:
        """Generate human-readable validation report"""
        
        report = []
        report.append("🔍 PRE-CONDITION VALIDATION REPORT")
        report.append("=" * 40)
        report.append("")
        
        # Overall status
        status_icon = "✅" if validation_report.overall_status == 'ready' else "⚠️" if validation_report.overall_status == 'needs_fixes' else "❌"
        report.append(f"{status_icon} OVERALL STATUS: {validation_report.overall_status.upper()}")
        report.append("")
        
        # Summary
        report.append("📊 SUMMARY")
        report.append("-" * 10)
        report.append(f"Total Checks: {validation_report.total_checks}")
        report.append(f"Passed: {validation_report.passed_checks}")
        report.append(f"Critical Failures: {validation_report.critical_failures}")
        report.append(f"Warnings: {validation_report.warnings}")
        report.append("")
        
        # Readiness assessment
        report.append("🎯 READINESS ASSESSMENT")
        report.append("-" * 22)
        mcp_icon = "✅" if validation_report.ready_for_mcp else "❌"
        cyclical_icon = "✅" if validation_report.ready_for_cyclical else "❌"
        report.append(f"{mcp_icon} MCP Ready: {validation_report.ready_for_mcp}")
        report.append(f"{cyclical_icon} Cyclical Optimization Ready: {validation_report.ready_for_cyclical}")
        report.append("")
        
        # Detailed results
        report.append("📋 DETAILED RESULTS")
        report.append("-" * 18)
        
        # Group by severity
        critical_results = [r for r in validation_report.results if r.severity == 'critical']
        warning_results = [r for r in validation_report.results if r.severity == 'warning']
        info_results = [r for r in validation_report.results if r.severity == 'info']
        
        if critical_results:
            report.append("🚨 CRITICAL ISSUES:")
            for result in critical_results:
                status_icon = "✅" if result.passed else "❌"
                report.append(f"  {status_icon} {result.name}")
                if not result.passed:
                    report.append(f"      {result.description}")
        
        if warning_results:
            report.append("")
            report.append("⚠️ WARNINGS:")
            for result in warning_results:
                status_icon = "✅" if result.passed else "⚠️"
                report.append(f"  {status_icon} {result.name}")
                if not result.passed:
                    report.append(f"      {result.description}")
        
        if info_results:
            report.append("")
            report.append("ℹ️ INFORMATION:")
            for result in info_results:
                status_icon = "✅" if result.passed else "ℹ️"
                report.append(f"  {status_icon} {result.name}")
        
        # Next steps
        if validation_report.next_steps:
            report.append("")
            report.append("🚀 NEXT STEPS")
            report.append("-" * 12)
            for i, step in enumerate(validation_report.next_steps, 1):
                report.append(f"  {i}. {step}")
        
        return "\n".join(report)

async def validate_project_readiness(project_root: str = ".") -> ValidationReport:
    """Main function to validate project readiness for cyclical MCP implementation"""
    
    validator = PreConditionValidator(project_root)
    validation_report = await validator.validate_all_preconditions()
    
    # Generate and display report
    report = validator.generate_report(validation_report)
    print(report)
    
    # Save detailed results
    with open("precondition_validation_report.json", 'w') as f:
        json.dump({
            "overall_status": validation_report.overall_status,
            "ready_for_mcp": validation_report.ready_for_mcp,
            "ready_for_cyclical": validation_report.ready_for_cyclical,
            "total_checks": validation_report.total_checks,
            "passed_checks": validation_report.passed_checks,
            "critical_failures": validation_report.critical_failures,
            "warnings": validation_report.warnings,
            "next_steps": validation_report.next_steps,
            "detailed_results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "description": r.description,
                    "severity": r.severity,
                    "details": r.details,
                    "fix_suggestions": r.fix_suggestions
                }
                for r in validation_report.results
            ]
        }, f, indent=2)
    
    logger.info("📊 Validation report saved to precondition_validation_report.json")
    
    return validation_report

if __name__ == "__main__":
    # Run validation
    validation_result = asyncio.run(validate_project_readiness())
    
    print("\n" + "="*50)
    print("🎯 VALIDATION SUMMARY")
    print("="*50)
    print(f"Overall Status: {validation_result.overall_status.upper()}")
    print(f"MCP Ready: {validation_result.ready_for_mcp}")
    print(f"Cyclical Optimization Ready: {validation_result.ready_for_cyclical}")
    print(f"Success Rate: {validation_result.passed_checks}/{validation_result.total_checks}")
    
    if validation_result.ready_for_cyclical:
        print("\n🎉 PROJECT IS READY FOR CYCLICAL MCP IMPLEMENTATION!")
    elif validation_result.ready_for_mcp:
        print("\n⚠️ Project is MCP-ready but needs cyclical optimization setup")
    else:
        print("\n❌ Project needs fixes before MCP implementation")
    
    print(f"\nNext steps: {len(validation_result.next_steps)} items to address")