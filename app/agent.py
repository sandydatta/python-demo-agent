# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from app.tools import (
    chroma_pdf_rag_search_tool,
    demo_ai_libraries_tool,
    demo_ml_libraries_tool,
    demo_python_features_tool,
    pdf_rag_search_tool,
)

root_agent = Agent(
    name="python_demo_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="""You are a helpful Python and AI ecosystem assistant.
You can execute demonstrations of standard Python features (collections, itertools, functools, typing, dataclasses, asyncio, concurrent.futures, pathlib, logging, json),
ML libraries (PyTorch, Scikit-Learn, NumPy, Matplotlib, Pandas), AI agent libraries (LangChain, LangGraph, LlamaIndex, ChromaDB, Pinecone, Sentence-Transformers, PEFT), and PDF RAG semantic search (Pinecone & ChromaDB).""",
    tools=[
        demo_python_features_tool,
        demo_ml_libraries_tool,
        demo_ai_libraries_tool,
        pdf_rag_search_tool,
        chroma_pdf_rag_search_tool,
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
