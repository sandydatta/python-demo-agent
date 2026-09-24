"""
Unit tests for python-demo-agent demo modules.
"""

from pathlib import Path
import pytest

from app.ai_agent_libraries_demo import run_all_ai_agent_libraries_demo
from app.ml_libraries_demo import run_all_ml_libraries_demo
from app.python_features_demo import run_all_python_features_demo
from app.tools import (
    demo_ai_libraries_tool,
    demo_ml_libraries_tool,
    demo_python_features_tool,
)


def test_python_features_demo(tmp_path: Path):
    res = run_all_python_features_demo(output_dir=tmp_path)
    assert res["fibonacci_cached(10)"] == 55
    assert res["partial_double(21)"] == 42
    assert "most_common_fruit" in res["collections_and_itertools"]
    assert res["pathlib_and_json"]["file_exists"] is True
    assert len(res["asyncio_results"]) == 3
    assert res["concurrent_futures_results"] == [0, 1, 4, 9, 16]


def test_ml_libraries_demo(tmp_path: Path):
    res = run_all_ml_libraries_demo(output_dir=tmp_path)
    assert res["numpy"]["array_shape"] == (2, 3)
    assert res["pandas"]["total_rows"] == 4
    assert Path(res["matplotlib_saved_plot"]).exists()
    assert "model_coef" in res["scikit_learn"]
    assert res["pytorch"]["input_shape"] == [2, 3]


def test_ai_agent_libraries_demo():
    res = run_all_ai_agent_libraries_demo()
    assert "Developer" in res["langchain"]["formatted_prompt"]
    assert "Completed!" in res["langgraph"]["graph_output"]
    assert res["llama_index"]["indexed_documents"] == 2
    assert res["chromadb"]["collection_name"] == "demo_collection"
    assert "Pinecone" in res["pinecone"]["status"]
    assert res["sentence_transformers"]["model_name"] == "all-MiniLM-L6-v2"
    assert res["peft"]["rank_r"] == 8


def test_tools():
    python_res = demo_python_features_tool()
    assert "fibonacci_cached(10)" in python_res

    ml_res = demo_ml_libraries_tool()
    assert "numpy" in ml_res

    ai_res = demo_ai_libraries_tool()
    assert "langchain" in ai_res
