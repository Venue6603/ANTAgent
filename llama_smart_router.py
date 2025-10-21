"""
Smart LLM Router - Automatically uses Ollama or llama-cpp based on availability
This module can replace direct llama-cpp imports with zero code changes.
"""

import os
import sys
import json
import requests
from typing import Dict, List, Optional, Any, Union


def check_ollama_available() -> bool:
    """Check if Ollama is available and has the required model."""
    try:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "")
        
        if not model:
            return False
            
        # Quick check if Ollama is running
        response = requests.get(f"{host}/api/version", timeout=1)
        return response.status_code == 200
    except:
        return False


def check_llama_cpp_available() -> bool:
    """Check if llama-cpp is available and model path exists."""
    try:
        import llama_cpp
        model_path = os.getenv("ANT_LLAMA_MODEL_PATH", "")
        return bool(model_path and os.path.exists(model_path))
    except ImportError:
        return False


class SmartLlama:
    """
    Smart router that automatically chooses between Ollama and llama-cpp.
    Provides the same interface as llama-cpp's Llama class.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_batch: int = 256,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        self.verbose = verbose
        self.n_ctx = n_ctx
        self._backend = None
        self._llm = None
        
        # Determine which backend to use
        use_ollama = os.getenv("FORCE_OLLAMA", "").lower() in ("true", "1", "yes")
        use_llama_cpp = os.getenv("FORCE_LLAMA_CPP", "").lower() in ("true", "1", "yes")
        
        if use_ollama or (not use_llama_cpp and check_ollama_available()):
            self._init_ollama(n_ctx, n_batch, verbose)
        elif use_llama_cpp or check_llama_cpp_available():
            self._init_llama_cpp(model_path, n_ctx, n_batch, n_gpu_layers, verbose, **kwargs)
        else:
            # Fallback: try ollama with default model
            if verbose:
                print("[SmartLlama] No backend explicitly available, trying Ollama with deepseek-coder-v2:16b")
            os.environ["OLLAMA_MODEL"] = "deepseek-coder-v2:16b"
            self._init_ollama(n_ctx, n_batch, verbose)
    
    def _init_ollama(self, n_ctx, n_batch, verbose):
        """Initialize Ollama backend."""
        self._backend = "ollama"
        self.model = os.getenv("OLLAMA_MODEL", "deepseek-coder-v2:16b")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        if verbose:
            print(f"[SmartLlama] Using Ollama backend")
            print(f"[SmartLlama] Model: {self.model}")
            print(f"[SmartLlama] Host: {self.ollama_host}")
    
    def _init_llama_cpp(self, model_path, n_ctx, n_batch, n_gpu_layers, verbose, **kwargs):
        """Initialize llama-cpp backend."""
        try:
            from ollama_adapter import Llama
            self._backend = "llama_cpp"
            
            # Use provided model_path or get from environment
            if not model_path:
                model_path = os.getenv("ANT_LLAMA_MODEL_PATH")
            
            self._llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
                **kwargs
            )
            
            if verbose:
                print(f"[SmartLlama] Using llama-cpp backend")
                print(f"[SmartLlama] Model: {model_path}")
        except Exception as e:
            if verbose:
                print(f"[SmartLlama] Failed to init llama-cpp: {e}")
            # Fall back to Ollama
            self._init_ollama(n_ctx, n_batch, verbose)
    
    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        repeat_penalty: float = 1.05,
        add_generation_prompt: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create chat completion using the selected backend.
        """
        if self._backend == "llama_cpp" and self._llm:
            # Use llama-cpp directly
            return self._llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                repeat_penalty=repeat_penalty,
                add_generation_prompt=add_generation_prompt,
                **kwargs
            )
        
        elif self._backend == "ollama":
            # Use Ollama API
            return self._ollama_chat_completion(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                repeat_penalty=repeat_penalty
            )
        
        else:
            # No backend available
            return {
                "choices": [{
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "no_backend"
                }]
            }
    
    def _ollama_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: Optional[List[str]],
        repeat_penalty: float
    ) -> Dict[str, Any]:
        """Handle Ollama API calls."""
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
            "messages": messages,
            "stream": False,
            "options": options
        }
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            
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
        except Exception as e:
            if self.verbose:
                print(f"[SmartLlama] Ollama error: {e}")
            return {
                "choices": [{
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "error"
                }]
            }


# Make it a drop-in replacement
Llama = SmartLlama


# For backward compatibility, expose the class under both names
__all__ = ['Llama', 'SmartLlama', 'check_ollama_available', 'check_llama_cpp_available']


if __name__ == "__main__":
    # Quick test
    print("Testing Smart LLM Router...")
    
    print(f"Ollama available: {check_ollama_available()}")
    print(f"llama-cpp available: {check_llama_cpp_available()}")
    
    llm = SmartLlama(verbose=True)
    
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, World!' in Python."}
        ],
        max_tokens=50
    )
    
    print("\nResponse:", result["choices"][0]["message"]["content"])
