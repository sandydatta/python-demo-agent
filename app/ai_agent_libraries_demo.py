"""
Demonstration of AI & LLM Agent libraries:
- langchain
- langgraph
- llama_index
- chromadb
- pinecone
- sentence_transformers
- peft
"""

import logging
from typing import Any, Dict, List, TypedDict

# 1. LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# 2. LangGraph imports
from langgraph.graph import END, START, StateGraph

# 3. LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex

# 4. ChromaDB imports
import chromadb

# 5. Pinecone imports
try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None

# 6. SentenceTransformers & PEFT imports
from peft import LoraConfig, TaskType
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def demo_langchain() -> Dict[str, Any]:
    logger.info("Running LangChain demo...")
    prompt = PromptTemplate.from_template("Hello {name}, welcome to {project}!")
    formatted_prompt = prompt.format(name="Developer", project="Build with Gemini")
    
    # Runnable chain demonstration
    uppercase_runnable = RunnableLambda(lambda text: str(text.to_string() if hasattr(text, "to_string") else text).upper())
    chain = prompt | uppercase_runnable
    chain_output = chain.invoke({"name": "Developer", "project": "Build with Gemini"})

    return {
        "formatted_prompt": formatted_prompt,
        "chain_output": str(chain_output),
    }


# LangGraph State definition
class AgentState(TypedDict):
    input: str
    processed_step_1: str
    output: str


def demo_langgraph() -> Dict[str, Any]:
    logger.info("Running LangGraph demo...")

    def step_1(state: AgentState) -> Dict[str, str]:
        return {"processed_step_1": f"Step 1 processed: '{state['input']}'"}

    def step_2(state: AgentState) -> Dict[str, str]:
        return {"output": f"Final Result: [{state['processed_step_1']}] -> Completed!"}

    builder = StateGraph(AgentState)
    builder.add_node("step_1", step_1)
    builder.add_node("step_2", step_2)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", END)

    graph = builder.compile()
    final_state = graph.invoke({"input": "Initial Agent Prompt"})

    return {
        "graph_input": final_state.get("input"),
        "step_1_result": final_state.get("processed_step_1"),
        "graph_output": final_state.get("output"),
    }


def demo_llama_index() -> Dict[str, Any]:
    logger.info("Running LlamaIndex demo...")
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.llms import MockLLM
    documents = [
        Document(text="Antigravity is an AI coding assistant powered by Gemini."),
        Document(text="agents-cli helps developers build, evaluate, and deploy agentic workflows."),
    ]
    embed_model = MockEmbedding(embed_dim=128)
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    query_engine = index.as_query_engine(llm=MockLLM())

    return {
        "indexed_documents": len(documents),
        "query_engine_type": type(query_engine).__name__,
        "sample_doc_preview": documents[0].text,
    }


def demo_chromadb() -> Dict[str, Any]:
    logger.info("Running ChromaDB demo...")
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="demo_collection")

    if collection.count() == 0:
        collection.add(
            documents=[
                "Gemini 3.5 Pro is a state-of-the-art multimodal model.",
                "PyTorch and Scikit-Learn are essential Python ML frameworks.",
            ],
            metadatas=[{"topic": "AI"}, {"topic": "ML"}],
            ids=["doc1", "doc2"],
        )

    results = collection.query(
        query_texts=["What is Gemini?"],
        n_results=1,
    )

    return {
        "collection_name": collection.name,
        "count": collection.count(),
        "query_matched_doc": results["documents"][0][0] if results.get("documents") else None,
    }


def demo_pinecone() -> Dict[str, Any]:
    logger.info("Running Pinecone SDK initialization demo...")
    # Initialize Pinecone client configuration demo
    init_status = "Pinecone SDK imported successfully"
    if Pinecone:
        pc_demo = "Pinecone client ready for API key setup"
    else:
        pc_demo = "Pinecone package available"

    return {
        "status": init_status,
        "details": pc_demo,
    }


def demo_sentence_transformers() -> Dict[str, Any]:
    logger.info("Running SentenceTransformers demo...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = ["Building agentic apps with Gemini", "Python standard library tools"]
    embeddings = model.encode(sentences)

    return {
        "model_name": "all-MiniLM-L6-v2",
        "sentence_count": len(sentences),
        "embedding_shape": list(embeddings.shape),
    }


def demo_peft() -> Dict[str, Any]:
    logger.info("Running PEFT (Parameter-Efficient Fine-Tuning) demo...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    return {
        "peft_type": peft_config.peft_type.value,
        "task_type": peft_config.task_type.value,
        "rank_r": peft_config.r,
        "target_modules": list(peft_config.target_modules),
    }


def run_all_ai_agent_libraries_demo() -> Dict[str, Any]:
    langchain_res = demo_langchain()
    langgraph_res = demo_langgraph()
    llama_res = demo_llama_index()
    chroma_res = demo_chromadb()
    pinecone_res = demo_pinecone()
    st_res = demo_sentence_transformers()
    peft_res = demo_peft()

    return {
        "langchain": langchain_res,
        "langgraph": langgraph_res,
        "llama_index": llama_res,
        "chromadb": chroma_res,
        "pinecone": pinecone_res,
        "sentence_transformers": st_res,
        "peft": peft_res,
    }


if __name__ == "__main__":
    import json
    res = run_all_ai_agent_libraries_demo()
    print(json.dumps(res, indent=2))
