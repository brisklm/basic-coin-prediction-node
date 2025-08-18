"""
API Key Testing Tool
===================
Use this script to test if your MCP API key is working correctly.
"""

import requests
import os
import json
from typing import Optional

def test_anthropic_api_key(api_key: Optional[str] = None) -> bool:
    """Test Anthropic API key"""
    
    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("❌ No Anthropic API key provided")
        return False
    
    print(f"🔍 Testing Anthropic API key: {api_key[:12]}...")
    
    try:
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Test with a simple message
        data = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Anthropic API key is VALID and working!")
            result = response.json()
            print(f"📝 Test response: {result.get('content', [{}])[0].get('text', 'OK')}")
            return True
        elif response.status_code == 401:
            print("❌ Anthropic API key is INVALID")
            print("🔧 Please check your API key and try again")
            return False
        else:
            print(f"⚠️ Unexpected response: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error testing Anthropic API: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing Anthropic API: {e}")
        return False

def test_openai_api_key(api_key: Optional[str] = None) -> bool:
    """Test OpenAI API key"""
    
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ No OpenAI API key provided")
        return False
    
    print(f"🔍 Testing OpenAI API key: {api_key[:12]}...")
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Test with a simple chat completion
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 10
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ OpenAI API key is VALID and working!")
            result = response.json()
            message = result.get('choices', [{}])[0].get('message', {}).get('content', 'OK')
            print(f"📝 Test response: {message}")
            return True
        elif response.status_code == 401:
            print("❌ OpenAI API key is INVALID")
            print("🔧 Please check your API key and try again")
            return False
        else:
            print(f"⚠️ Unexpected response: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error testing OpenAI API: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing OpenAI API: {e}")
        return False

def interactive_api_key_setup():
    """Interactive setup for API keys"""
    
    print("🔑 API KEY TESTING TOOL")
    print("=" * 25)
    print()
    print("This tool helps you test your MCP API keys for the competition toolkit.")
    print("API keys enable AI-powered optimization for better performance!")
    print()
    
    print("🎯 Choose your API provider:")
    print("1. 💎 Anthropic Claude (Recommended - better for competitions)")
    print("2. 🤖 OpenAI GPT (Alternative)")
    print("3. 🔍 Test both")
    print("4. ❌ Skip API setup (use basic mode)")
    print()
    
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            # Test Anthropic
            print("\n💎 ANTHROPIC CLAUDE API SETUP")
            print("-" * 30)
            print("Get your API key at: https://console.anthropic.com/")
            print("Your key should start with: sk-ant-")
            print()
            
            api_key = input("Enter your Anthropic API key: ").strip()
            if api_key:
                success = test_anthropic_api_key(api_key)
                if success:
                    print("\n🎉 Perfect! Your Anthropic API key is ready!")
                    print("💡 Use it in your code like this:")
                    print(f'   mcp_api_key="{api_key}"')
                    return api_key
                else:
                    print("\n🔧 Please check your API key and try again.")
            break
            
        elif choice == "2":
            # Test OpenAI
            print("\n🤖 OPENAI GPT API SETUP")
            print("-" * 25)
            print("Get your API key at: https://platform.openai.com/")
            print("Your key should start with: sk-")
            print()
            
            api_key = input("Enter your OpenAI API key: ").strip()
            if api_key:
                success = test_openai_api_key(api_key)
                if success:
                    print("\n🎉 Perfect! Your OpenAI API key is ready!")
                    print("💡 Use it in your code like this:")
                    print(f'   mcp_api_key="{api_key}"')
                    return api_key
                else:
                    print("\n🔧 Please check your API key and try again.")
            break
            
        elif choice == "3":
            # Test both
            print("\n🔍 TESTING BOTH API PROVIDERS")
            print("-" * 32)
            
            anthropic_key = input("Enter Anthropic API key (or press Enter to skip): ").strip()
            openai_key = input("Enter OpenAI API key (or press Enter to skip): ").strip()
            
            anthropic_works = False
            openai_works = False
            
            if anthropic_key:
                print("\n💎 Testing Anthropic...")
                anthropic_works = test_anthropic_api_key(anthropic_key)
            
            if openai_key:
                print("\n🤖 Testing OpenAI...")
                openai_works = test_openai_api_key(openai_key)
            
            if anthropic_works:
                print(f"\n🏆 Recommendation: Use Anthropic (better for competitions)")
                return anthropic_key
            elif openai_works:
                print(f"\n✅ OpenAI is working and ready to use!")
                return openai_key
            else:
                print("\n😞 No working API keys found.")
            break
            
        elif choice == "4":
            print("\n📝 No problem! You can use the toolkit without API keys.")
            print("💡 You'll still get great performance, just without AI optimization.")
            print("🚀 You can always add an API key later for enhanced performance!")
            return None
            
        else:
            print("❌ Please enter 1, 2, 3, or 4")

def main():
    """Main function"""
    
    # Check for environment variables first
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if anthropic_key or openai_key:
        print("🔍 Found API keys in environment variables!")
        print()
        
        if anthropic_key:
            print("💎 Testing Anthropic API key from environment...")
            if test_anthropic_api_key(anthropic_key):
                print("✅ Your environment is ready!")
                return
        
        if openai_key:
            print("🤖 Testing OpenAI API key from environment...")
            if test_openai_api_key(openai_key):
                print("✅ Your environment is ready!")
                return
        
        print("⚠️ Environment API keys not working. Let's set up manually...")
        print()
    
    # Interactive setup
    api_key = interactive_api_key_setup()
    
    if api_key:
        print("\n🎯 NEXT STEPS:")
        print("1. Save your API key securely")
        print("2. Use it in your competition code:")
        print("   autonomous_competition_solution(")
        print("       'https://www.kaggle.com/competitions/titanic',")
        print(f"       mcp_api_key='{api_key}'")
        print("   )")
        print("3. Enjoy enhanced AI-powered optimization!")
    else:
        print("\n🎯 NEXT STEPS:")
        print("1. Use the toolkit in basic mode (still great performance!)")
        print("2. Get an API key later if you want enhanced features")
        print("3. Start competing!")
    
    print("\n🏆 Happy competing!")

if __name__ == "__main__":
    main()