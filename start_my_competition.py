"""
Start My Competition - Easy Template
===================================
Just replace the competition URL and run this script!
"""

from enhanced_competition_toolkit import autonomous_competition_solution_simple
from cyclical_mcp_system import CyclicalConfig
import sys
import time
from pathlib import Path

def get_user_competition_choice():
    """Let user choose a competition or enter their own URL"""
    
    print("🎯 AI COMPETITION TOOLKIT - REAL COMPETITION SOLVER")
    print("=" * 55)
    print()
    print("Choose how you want to proceed:")
    print()
    
    # Popular competition options
    popular_competitions = [
        {
            "name": "Titanic - Machine Learning from Disaster", 
            "url": "https://www.kaggle.com/competitions/titanic",
            "type": "Binary Classification",
            "difficulty": "Beginner",
            "expected_time": "5-8 minutes"
        },
        {
            "name": "House Prices - Advanced Regression",
            "url": "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques", 
            "type": "Regression",
            "difficulty": "Intermediate",
            "expected_time": "8-12 minutes"
        },
        {
            "name": "Spaceship Titanic",
            "url": "https://www.kaggle.com/competitions/spaceship-titanic",
            "type": "Binary Classification", 
            "difficulty": "Beginner",
            "expected_time": "6-10 minutes"
        }
    ]
    
    print("📋 POPULAR COMPETITIONS (recommended for first try):")
    for i, comp in enumerate(popular_competitions, 1):
        print(f"  {i}. {comp['name']}")
        print(f"     Type: {comp['type']} | Difficulty: {comp['difficulty']}")
        print(f"     Expected time: {comp['expected_time']}")
        print(f"     URL: {comp['url']}")
        print()
    
    print(f"  {len(popular_competitions) + 1}. Enter my own competition URL")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(popular_competitions):
                selected_comp = popular_competitions[choice_num - 1]
                print(f"\n✅ Selected: {selected_comp['name']}")
                return selected_comp['url']
            elif choice_num == len(popular_competitions) + 1:
                url = input("\n📝 Enter your competition URL: ").strip()
                if url:
                    print(f"\n✅ Using your URL: {url}")
                    return url
                else:
                    print("❌ Please enter a valid URL")
            else:
                print(f"❌ Please enter a number between 1 and {len(popular_competitions) + 1}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def get_optimization_level():
    """Let user choose optimization level"""
    
    print("\n🚀 OPTIMIZATION LEVEL:")
    print("-" * 25)
    print("1. 🏃 BASIC - Fast & Simple (5-10 minutes)")
    print("   • Automatic everything")
    print("   • Good performance")
    print("   • No API keys required")
    print()
    print("2. 🧠 ENHANCED - Better Performance (8-15 minutes)")
    print("   • AI-powered optimization")
    print("   • +2-3% better performance")
    print("   • Requires API key (Claude or ChatGPT)")
    print()
    print("3. 🔄 MAXIMUM - Best Performance (15-30 minutes)")
    print("   • Cyclical MCP optimization")
    print("   • +3-5% better performance")
    print("   • Requires API key (Claude or ChatGPT)")
    print()
    
    while True:
        try:
            choice = input("Choose optimization level (1-3): ").strip()
            choice_num = int(choice)
            
            if choice_num in [1, 2, 3]:
                return choice_num
            else:
                print("❌ Please enter 1, 2, or 3")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def get_api_provider_and_key(optimization_level):
    """Get API provider choice and key if enhanced optimization is selected"""
    
    if optimization_level == 1:
        return None, None
    
    print(f"\n🔑 API KEY REQUIRED FOR ENHANCED OPTIMIZATION:")
    print("-" * 45)
    print("Choose your AI provider for optimization:")
    print()
    print("1. 💎 Anthropic Claude (Recommended)")
    print("   • Better for competition optimization")
    print("   • Lower cost (~$0.10-$1.00 per competition)")
    print("   • Get key at: https://console.anthropic.com/")
    print("   • Key format: sk-ant-...")
    print()
    print("2. 🤖 OpenAI ChatGPT")
    print("   • Alternative AI provider")
    print("   • Higher cost (~$0.20-$2.00 per competition)")
    print("   • Get key at: https://platform.openai.com/")
    print("   • Key format: sk-...")
    print()
    print("3. ❌ Skip (Use Basic mode)")
    print("   • No API key needed")
    print("   • Still gets good performance")
    print()
    
    while True:
        try:
            choice = input("Choose your API provider (1-3): ").strip()
            choice_num = int(choice)
            
            if choice_num == 1:
                # Anthropic Claude
                print("\n💎 ANTHROPIC CLAUDE SETUP:")
                print("-" * 25)
                api_key = input("Enter your Anthropic API key (sk-ant-...): ").strip()
                if api_key:
                    if api_key.startswith("sk-ant-"):
                        print("✅ Anthropic API key format looks correct!")
                        return "anthropic", api_key
                    else:
                        print("⚠️ Warning: Anthropic keys usually start with 'sk-ant-'")
                        return "anthropic", api_key
                else:
                    print("📝 No API key provided. Using Basic mode.")
                    return None, None
                    
            elif choice_num == 2:
                # OpenAI ChatGPT
                print("\n🤖 OPENAI CHATGPT SETUP:")
                print("-" * 25)
                api_key = input("Enter your OpenAI API key (sk-...): ").strip()
                if api_key:
                    if api_key.startswith("sk-"):
                        print("✅ OpenAI API key format looks correct!")
                        return "openai", api_key
                    else:
                        print("⚠️ Warning: OpenAI keys usually start with 'sk-'")
                        return "openai", api_key
                else:
                    print("📝 No API key provided. Using Basic mode.")
                    return None, None
                    
            elif choice_num == 3:
                # Skip API key
                print("📝 Using Basic optimization mode (still great performance!)")
                return None, None
                
            else:
                print("❌ Please enter 1, 2, or 3")
                
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def ask_optional_repo_urls():
    """Prompt user to optionally provide one or more GitHub repo URLs (comma-separated)."""
    try:
        print("\n🔗 (Optional) Provide GitHub repository URL(s) with prior solutions")
        print("   Examples: https://github.com/user/repo or multiple comma-separated")
        ans = input("Repo URL(s) [press Enter to skip]: ").strip()
        if not ans:
            return []
        return [u.strip() for u in ans.split(',') if u.strip()]
    except KeyboardInterrupt:
        return []

def solve_competition(competition_url, optimization_level, api_provider=None, api_key=None):
    """Solve the competition with specified parameters"""
    
    print(f"\n🚀 STARTING COMPETITION SOLUTION")
    print("=" * 35)
    print(f"📍 Competition: {competition_url}")
    print(f"⚡ Optimization: Level {optimization_level}")
    
    if api_provider and api_key:
        provider_name = "Anthropic Claude" if api_provider == "anthropic" else "OpenAI ChatGPT"
        print(f"🤖 AI Provider: {provider_name}")
        print(f"🔑 API Key: Provided")
    else:
        print(f"🔑 API Key: Not provided (Basic mode)")
    print()
    
    # Create output directory
    output_dir = Path("./competition_results")
    output_dir.mkdir(exist_ok=True)
    
    print("⏳ Processing... This may take several minutes.")
    print("   You can watch the progress in the console output below.")
    print()
    
    start_time = time.time()
    
    try:
        if optimization_level == 1:
            # Basic mode
            results = autonomous_competition_solution_simple(
                competition_url=competition_url,
                output_dir=str(output_dir)
            )
        elif optimization_level == 2:
            # Enhanced mode
            repo_urls = ask_optional_repo_urls()
            results = autonomous_competition_solution_simple(
                competition_url=competition_url,
                mcp_api_key=api_key,
                api_provider=api_provider,
                output_dir=str(output_dir),
                repo_urls=repo_urls
            )
        else:
            # Maximum mode with cyclical optimization
            repo_urls = ask_optional_repo_urls()
            results = autonomous_competition_solution_simple(
                competition_url=competition_url,
                mcp_api_key=api_key,
                api_provider=api_provider,
                enable_cyclical_optimization=True,
                cyclical_config=CyclicalConfig(
                    max_iterations=10,
                    no_improvement_threshold=3
                ),
                output_dir=str(output_dir),
                repo_urls=repo_urls
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\n🎊 COMPETITION SOLVED!")
        print("=" * 35)
        print(f"⏱️  Processing Time: {processing_time/60:.1f} minutes")

        # Handle both demo and real solution report formats
        if isinstance(results, dict) and 'model_performance' in results:
            print(f"📊 Final Score: {results['model_performance']['final_score']:.4f}")
            print(f"🏆 Best Model: {results['model_performance']['best_single_model']['name']}")
            if 'best_ensemble' in results['model_performance']:
                print(f"🤝 Best Ensemble: {results['model_performance']['best_ensemble']['name']}")
        elif isinstance(results, dict) and 'training_report' in results:
            tr = results['training_report']
            models_list = tr.get('models_trained', [])
            print(f"🧠 Models trained: {', '.join(models_list) if models_list else 'n/a'}")
            if tr.get('optimization_applied'):
                print("🛠️  Optimization: enabled")

        print(f"📁 Files saved to: {output_dir.absolute()}")
        submission_path = results.get('files_generated', {}).get('submission') if isinstance(results, dict) else None
        if submission_path:
            print(f"📤 Submission file: {submission_path}")
        
        # List generated files
        print("\n📋 Generated Files:")
        files_generated = results.get('files_generated', {}) if isinstance(results, dict) else {}
        for file_type, filename in files_generated.items():
            candidate = Path(filename)
            file_path = candidate if candidate.is_absolute() else (output_dir / filename)
            if file_path.exists():
                print(f"  ✅ {file_type.title()}: {file_path}")
            else:
                print(f"  ❌ {file_type.title()}: {file_path} (not found)")
        
        print("\n🚀 Next Steps:")
        print("1. Review the generated submission file")
        print("2. Submit it to the competition platform")
        print("3. Check your leaderboard position!")
        print("4. Try enhanced optimization for better performance")
        
        return results
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n❌ ERROR OCCURRED AFTER {processing_time/60:.1f} MINUTES:")
        print(f"   {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify the competition URL is correct")
        print("3. Try a simpler competition (like Titanic)")
        print("4. Check if all dependencies are installed")
        print("\n💡 Try running this first:")
        print("   python -c \"import numpy, pandas, sklearn; print('Dependencies OK')\"")
        
        return None

def main():
    """Main function"""
    
    try:
        # Get user's competition choice
        competition_url = get_user_competition_choice()
        
        # Get optimization level
        optimization_level = get_optimization_level()
        
        # Get API provider and key if needed
        api_provider, api_key = get_api_provider_and_key(optimization_level)
        
        # Solve the competition
        results = solve_competition(competition_url, optimization_level, api_provider, api_key)
        
        if results:
            print("\n🏆 SUCCESS! Your competition solution is ready!")
        else:
            print("\n😞 Something went wrong. Please try again.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Process interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()