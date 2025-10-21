#!/usr/bin/env python3
"""
Test script to verify Ollama integration with DeepSeek Coder v2
"""

import os
import sys
import json
import requests

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ollama_connection():
    """Test if Ollama is running and model is available"""
    print("=" * 60)
    print("Testing Ollama Connection")
    print("=" * 60)
    
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "deepseek-coder-v2:16b")
    
    print(f"Host: {host}")
    print(f"Model: {model}")
    
    try:
        # Check Ollama version
        response = requests.get(f"{host}/api/version", timeout=5)
        if response.status_code == 200:
            version = response.json()
            print(f"✓ Ollama is running (version: {version.get('version', 'unknown')})")
        else:
            print("✗ Ollama is not responding")
            return False
            
        # Check if model exists
        response = requests.post(
            f"{host}/api/show",
            json={"name": model},
            timeout=5
        )
        if response.status_code == 200:
            print(f"✓ Model {model} is available")
            model_info = response.json()
            if "details" in model_info:
                details = model_info["details"]
                print(f"  - Format: {details.get('format', 'unknown')}")
                print(f"  - Parameter size: {details.get('parameter_size', 'unknown')}")
        else:
            print(f"✗ Model {model} not found")
            print(f"  Run: ollama pull {model}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama")
        print("  Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True


def test_adapter():
    """Test the Ollama adapter"""
    print("\n" + "=" * 60)
    print("Testing Ollama Adapter")
    print("=" * 60)
    
    try:
        from ollama_adapter import OllamaAdapter, check_ollama_connection
        
        # Check connection
        connected, message = check_ollama_connection()
        print(f"Adapter connection check: {message}")
        
        if not connected:
            print("✗ Cannot proceed without connection")
            return False
        
        # Create adapter instance
        print("\nCreating adapter instance...")
        adapter = OllamaAdapter(
            n_ctx=2048,
            verbose=True
        )
        
        # Test simple completion
        print("\nTesting chat completion...")
        
        test_messages = [
            {"role": "system", "content": "You are a code assistant. Be concise."},
            {"role": "user", "content": "Write a Python function to reverse a string. Just the function, no explanation."}
        ]
        
        result = adapter.create_chat_completion(
            messages=test_messages,
            temperature=0.1,
            max_tokens=100
        )
        
        if result and "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            print("\n✓ Adapter test successful!")
            print(f"Response preview: {content[:200]}...")
            
            # Verify it looks like Python code
            if "def " in content or "return" in content:
                print("✓ Response appears to be valid Python code")
            else:
                print("⚠ Response may not be code as expected")
            
            return True
        else:
            print("✗ No valid response from adapter")
            return False
            
    except ImportError as e:
        print(f"✗ Cannot import adapter: {e}")
        print("  Make sure ollama_adapter.py is in the same directory")
        return False
    except Exception as e:
        print(f"✗ Adapter test failed: {e}")
        return False


def test_code_generation():
    """Test actual code generation capability"""
    print("\n" + "=" * 60)
    print("Testing Code Generation Quality")
    print("=" * 60)
    
    try:
        from ollama_adapter import OllamaAdapter
        
        adapter = OllamaAdapter(n_ctx=4096, verbose=False)
        
        # Test diff generation (similar to what your app does)
        test_prompt = """Generate a unified diff to add a new function called 'calculate_sum' to a Python file.
The function should take a list of numbers and return their sum.

Output ONLY a valid unified diff starting with 'diff --git'.
No explanations, no markdown."""

        result = adapter.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a diff generator. Output only valid unified diffs."},
                {"role": "user", "content": test_prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        if result and "choices" in result:
            content = result["choices"][0]["message"]["content"]
            
            # Check if it looks like a diff
            if "diff --git" in content or "@@ " in content:
                print("✓ Generated output appears to be a diff")
                print("\nGenerated diff preview:")
                print("-" * 40)
                lines = content.split('\n')[:10]
                for line in lines:
                    print(line)
                if len(content.split('\n')) > 10:
                    print("...")
                print("-" * 40)
                return True
            else:
                print("✗ Output doesn't look like a unified diff")
                print(f"Got: {content[:200]}...")
                return False
        
    except Exception as e:
        print(f"✗ Code generation test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("DeepSeek Coder v2 Ollama Integration Test Suite")
    print("=" * 60)
    
    # Set environment if not already set
    if not os.getenv("OLLAMA_MODEL"):
        os.environ["OLLAMA_MODEL"] = "deepseek-coder-v2:16b"
        print(f"Set OLLAMA_MODEL={os.environ['OLLAMA_MODEL']}")
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Adapter Functionality", test_adapter),
        ("Code Generation", test_code_generation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! The integration is ready to use.")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
