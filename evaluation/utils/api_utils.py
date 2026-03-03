"""
API Utilities

Functions for calling vLLM and Judge APIs.
"""

import json
import time
import random
import requests
import threading
from typing import Dict, List, Any, Optional, Callable
from openai import OpenAI


# Thread-local storage for API clients
_thread_local = threading.local()


def get_openai_client(api_key: str, base_url: str) -> OpenAI:
    """Get or create thread-local OpenAI client."""
    key = f"{api_key}_{base_url}"
    if not hasattr(_thread_local, "clients"):
        _thread_local.clients = {}
    if key not in _thread_local.clients:
        _thread_local.clients[key] = OpenAI(api_key=api_key, base_url=base_url)
    return _thread_local.clients[key]


def call_vllm(
    messages: List[Dict],
    model_id: str,
    api_url: str = "http://localhost:8000/v1/chat/completions",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    top_k: int = -1,
    timeout: int = 120,
) -> str:
    """
    Call vLLM server with chat messages.
    
    Args:
        messages: List of message dicts with role and content
        model_id: Model identifier
        api_url: vLLM API endpoint
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling (-1 to disable)
        timeout: Request timeout in seconds
    
    Returns:
        Generated response text
    """
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    if top_k > 0:
        payload["top_k"] = top_k
    
    response = requests.post(
        api_url,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]


def call_judge_api(
    messages: List[Dict],
    model: str = "gpt-5-nano",
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 180,
) -> str:
    """
    Call Judge API (OpenAI-compatible) for evaluation.
    
    Args:
        messages: List of message dicts
        model: Judge model name
        api_key: API key
        base_url: API base URL
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        timeout: Request timeout
    
    Returns:
        Judge response text
    """
    client = get_openai_client(api_key, base_url)
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    
    return response.choices[0].message.content


def call_with_retries(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    parse_json: bool = True,
    **kwargs,
) -> Any:
    """
    Call function with exponential backoff retries.
    
    Args:
        fn: Function to call
        max_retries: Maximum retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay cap
        parse_json: Whether to parse response as JSON
        **kwargs: Arguments to pass to function
    
    Returns:
        Function result (parsed as JSON if parse_json=True)
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = fn(**kwargs)
            
            if parse_json:
                if isinstance(result, str):
                    if not result.strip():
                        raise ValueError("Empty response")
                    result = json.loads(result)
                elif not isinstance(result, dict):
                    raise TypeError(f"Unexpected type: {type(result)}")
            
            return result
            
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = min(max_delay, base_delay * (2 ** attempt))
                delay *= (0.5 + random.random())  # Jitter
                time.sleep(delay)
    
    raise last_error


def check_vllm_health(host: str = "localhost", port: int = 8000) -> bool:
    """Check if vLLM server is healthy."""
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False
