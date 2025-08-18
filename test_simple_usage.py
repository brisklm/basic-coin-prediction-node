"""
Test Simple Usage - Both API Providers
======================================
Test the new simple autonomous competition solution with different configurations
"""

from enhanced_competition_toolkit import autonomous_competition_solution_simple
from cyclical_mcp_system import CyclicalConfig
import time

def test_basic_mode():
    """Test basic mode without API key"""
    
    print("🧪 TEST 1: BASIC MODE (No API Key)")
    print("=" * 40)
    print("Testing the system without any API key...")
    print()
    
    try:
        results = autonomous_competition_solution_simple(
            competition_url="https://www.kaggle.com/competitions/titanic",
            output_dir="./test_basic_results"
        )
        
        print("✅ Basic mode test completed!")
        print(f"📊 Final Score: {results['model_performance']['final_score']}")
        print(f"🏆 Best Model: {results['model_performance']['best_single_model']['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic mode test failed: {e}")
        return False

def test_claude_api_simulation():
    """Test Claude API simulation"""
    
    print("\n🧪 TEST 2: CLAUDE API SIMULATION")
    print("=" * 35)
    print("Testing with simulated Claude API key...")
    print("(This would use real Claude API if you provide a real key)")
    print()
    
    try:
        results = autonomous_competition_solution_simple(
            competition_url="https://www.kaggle.com/competitions/titanic",
            mcp_api_key="sk-ant-demo_key_for_testing",  # Demo key
            api_provider="anthropic",
            output_dir="./test_claude_results"
        )
        
        print("✅ Claude API simulation completed!")
        print(f"📊 Final Score: {results['model_performance']['final_score']}")
        print(f"🏆 Best Model: {results['model_performance']['best_single_model']['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Claude API test failed: {e}")
        return False

def test_chatgpt_api_simulation():
    """Test ChatGPT API simulation"""
    
    print("\n🧪 TEST 3: CHATGPT API SIMULATION")
    print("=" * 36)
    print("Testing with simulated ChatGPT API key...")
    print("(This would use real ChatGPT API if you provide a real key)")
    print()
    
    try:
        results = autonomous_competition_solution_simple(
            competition_url="https://www.kaggle.com/competitions/house-prices",
            mcp_api_key="sk-demo_openai_key_for_testing",  # Demo key
            api_provider="openai",
            enable_cyclical_optimization=True,
            cyclical_config=CyclicalConfig(max_iterations=3),
            output_dir="./test_chatgpt_results"
        )
        
        print("✅ ChatGPT API simulation completed!")
        print(f"📊 Final Score: {results['model_performance']['final_score']}")
        print(f"🏆 Best Model: {results['model_performance']['best_single_model']['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ ChatGPT API test failed: {e}")
        return False

def show_usage_examples():
    """Show usage examples for both API providers"""
    
    print("\n📖 USAGE EXAMPLES")
    print("=" * 18)
    print()
    
    print("💎 CLAUDE (ANTHROPIC) USAGE:")
    print("-" * 30)
    print("from enhanced_competition_toolkit import autonomous_competition_solution_simple")
    print()
    print("results = autonomous_competition_solution_simple(")
    print('    competition_url="https://www.kaggle.com/competitions/titanic",')
    print('    mcp_api_key="sk-ant-your_claude_key_here",')
    print('    api_provider="anthropic",')
    print('    enable_cyclical_optimization=True')
    print(")")
    print()
    
    print("🤖 CHATGPT (OPENAI) USAGE:")
    print("-" * 28)
    print("from enhanced_competition_toolkit import autonomous_competition_solution_simple")
    print()
    print("results = autonomous_competition_solution_simple(")
    print('    competition_url="https://www.kaggle.com/competitions/titanic",')
    print('    mcp_api_key="sk-your_openai_key_here",')
    print('    api_provider="openai",')
    print('    enable_cyclical_optimization=True')
    print(")")
    print()
    
    print("🏃 BASIC MODE (NO API KEY):")
    print("-" * 28)
    print("from enhanced_competition_toolkit import autonomous_competition_solution_simple")
    print()
    print("results = autonomous_competition_solution_simple(")
    print('    competition_url="https://www.kaggle.com/competitions/titanic"')
    print(")")
    print()

def show_api_key_info():
    """Show information about getting API keys"""
    
    print("🔑 API KEY INFORMATION")
    print("=" * 23)
    print()
    
    print("💎 ANTHROPIC CLAUDE:")
    print("   🌐 Website: https://console.anthropic.com/")
    print("   🔑 Key format: sk-ant-...")
    print("   💰 Cost: ~$0.10-$1.00 per competition")
    print("   ⭐ Recommended for competitions")
    print()
    
    print("🤖 OPENAI CHATGPT:")
    print("   🌐 Website: https://platform.openai.com/")
    print("   🔑 Key format: sk-...")
    print("   💰 Cost: ~$0.20-$2.00 per competition")
    print("   ✅ Good alternative option")
    print()
    
    print("💡 TESTING API KEYS:")
    print("   Run: python test_api_key.py")
    print("   This will help you verify your keys work correctly.")
    print()

def main():
    """Main test function"""
    
    print("🚀 AI COMPETITION TOOLKIT - API PROVIDER TESTING")
    print("=" * 55)
    print()
    print("This script tests the new dual API provider support!")
    print("You can now use either Claude or ChatGPT for AI optimization.")
    print()
    
    # Show API key information
    show_api_key_info()
    
    # Show usage examples
    show_usage_examples()
    
    # Run tests
    print("🧪 RUNNING TESTS...")
    print()
    
    tests_passed = 0
    total_tests = 3
    
    # Test basic mode
    if test_basic_mode():
        tests_passed += 1
    
    # Test Claude simulation
    if test_claude_api_simulation():
        tests_passed += 1
    
    # Test ChatGPT simulation
    if test_chatgpt_api_simulation():
        tests_passed += 1
    
    print(f"\n📊 TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Both API providers are ready to use!")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Get an API key (Claude or ChatGPT)")
    print("2. Test it with: python test_api_key.py")
    print("3. Run a real competition: python start_my_competition.py")
    print("4. Submit your results and climb the leaderboard!")
    print()
    print("🏆 Happy competing!")

if __name__ == "__main__":
    main()