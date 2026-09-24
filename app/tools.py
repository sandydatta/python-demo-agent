"""
ADK Tools exposing Python Standard Library, ML, and AI Framework demos.
"""

from typing import Any, Dict
from app.python_features_demo import run_all_python_features_demo
from app.ml_libraries_demo import run_all_ml_libraries_demo
from app.ai_agent_libraries_demo import run_all_ai_agent_libraries_demo


def demo_python_features_tool() -> Dict[str, Any]:
    """Runs examples of collections, itertools, functools, typing, dataclasses, asyncio, concurrent.futures, pathlib, logging, json."""
    return run_all_python_features_demo()


def demo_ml_libraries_tool() -> Dict[str, Any]:
    """Runs examples of PyTorch, Scikit-Learn, NumPy, Matplotlib, and Pandas."""
    return run_all_ml_libraries_demo()


def demo_ai_libraries_tool() -> Dict[str, Any]:
    """Runs examples of LangChain, LangGraph, LlamaIndex, Chroma, Pinecone, Sentence-Transformers, and PEFT."""
    return run_all_ai_agent_libraries_demo()
