"""
Ollama adapter for DeepSeek Coder v2 - Drop-in replacement for llama-cpp
This adapter provides a compatible interface to use Ollama models with minimal code changes.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any


class OllamaAdapter:
    """
    Adapter class that mimics llama-cpp's Llama interface but uses Ollama backend.
    Provides the same API as llama-cpp for minimal code changes.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,  # Ignored, kept for compatibility
        n_ctx: int = 4096,
        n_batch: int = 256,
        n_gpu_layers: int = -1,  # Ignored, Ollama handles this
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize Ollama adapter with similar parameters to llama-cpp.
        
        Args:
            model_path: Ignored, model name is set via environment variable
            n_ctx: Context window size (maps to num_ctx in Ollama)
            n_batch: Ignored, Ollama handles batching internally
            n_gpu_layers: Ignored, Ollama handles GPU acceleration
            verbose: Enable verbose logging
        """
        self.model = os.getenv("OLLAMA_MODEL", "deepseek-coder-v2:16b")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.n_ctx = n_ctx
        self.verbose = verbose
        
        if self.verbose:
            print(f"[OllamaAdapter] Initialized with model: {self.model}")
            print(f"[OllamaAdapter] Host: {self.ollama_host}")
            print(f"[OllamaAdapter] Context size: {self.n_ctx}")
    
    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        repeat_penalty: float = 1.05,
        add_generation_prompt: bool = True,  # Ignored, for compatibility
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create chat completion using Ollama API.
        Mimics llama-cpp's create_chat_completion interface.
        
        Returns:
            Dictionary matching llama-cpp response format
        """
        # Prepare the Ollama API request
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        # Ollama API parameters
        options = {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "num_ctx": self.n_ctx,
            "repeat_penalty": repeat_penalty,
        }
        
        if stop:
            options["stop"] = stop
        
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": options
        }
        
        if self.verbose:
            print(f"[OllamaAdapter] Sending request to Ollama...")
            print(f"[OllamaAdapter] Model: {self.model}")
            print(f"[OllamaAdapter] Temperature: {temperature}, Top-p: {top_p}")
        
        try:
            # Make request to Ollama
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            
            if self.verbose:
                print(f"[OllamaAdapter] Response received")
                if "eval_count" in result:
                    print(f"[OllamaAdapter] Tokens generated: {result.get('eval_count', 0)}")
            
            # Format response to match llama-cpp structure
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": result.get("message", {}).get("content", "")
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                }
            }
            
        except requests.exceptions.Timeout:
            if self.verbose:
                print("[OllamaAdapter] Request timed out")
            return {
                "choices": [{
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "timeout"
                }]
            }
        except Exception as e:
            if self.verbose:
                print(f"[OllamaAdapter] Error: {e}")
            return {
                "choices": [{
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "error"
                }]
            }


# Alias for drop-in replacement
Llama = OllamaAdapter


def check_ollama_connection():
    """
    Check if Ollama is running and the model is available.
    """
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "deepseek-coder-v2:16b")
    
    try:
        # Check if Ollama is running
        response = requests.get(f"{host}/api/version", timeout=5)
        if response.status_code != 200:
            return False, "Ollama is not responding"
        
        # Check if model exists
        response = requests.post(
            f"{host}/api/show",
            json={"name": model},
            timeout=5
        )
        if response.status_code == 200:
            return True, f"Model {model} is available"
        else:
            return False, f"Model {model} not found. Run: ollama pull {model}"
            
    except Exception as e:
        return False, f"Cannot connect to Ollama: {e}"


if __name__ == "__main__":
    # Test the adapter
    print("Testing Ollama Adapter...")
    
    # Check connection
    connected, message = check_ollama_connection()
    print(f"Connection check: {message}")
    
    if connected:
        # Test chat completion
        llm = OllamaAdapter(verbose=True)
        
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a Python function to calculate fibonacci numbers."}
            ],
            temperature=0.2,
            max_tokens=200
        )
        
        print("\nResponse:")
        print(result["choices"][0]["message"]["content"])
