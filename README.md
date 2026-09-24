# Python Demo Agent (`python-demo-agent`)

A comprehensive demonstration Python project built with `agents-cli` and ADK, featuring standard library features, popular machine learning libraries, and modern AI/LLM agent frameworks.

---

## 🚀 How to Run Locally

### Prerequisites

- Python `>= 3.11`
- [`uv`](https://docs.astral.sh/uv/) (Python package manager)
- [`agents-cli`](https://google.github.io/agents-cli/) (`uv tool install google-agents-cli`)

### Setup Instructions

1. **Clone & navigate to the repository:**
   ```bash
   cd /config/.gemini/antigravity/scratch/python-demo-agent
   ```

2. **Create the virtual environment & install dependencies:**
   ```bash
   # Create a Python 3.12 virtual environment
   uv venv --python 3.12

   # Install CPU PyTorch
   uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu

   # Install remaining ML & AI dependencies
   uv pip install --python .venv scikit-learn numpy matplotlib pandas langchain langchain-community langgraph llama-index chromadb pinecone sentence-transformers peft pytest

   # Install the project in editable mode
   uv pip install --python .venv -e .
   ```

### Running Demos Locally

Execute the main CLI runner [`app/main.py`](file:///config/.gemini/antigravity/scratch/python-demo-agent/app/main.py) to run all demos and view structured JSON results:

```bash
# Run all demonstrations (Python Stdlib, ML, AI Frameworks)
.venv/bin/python -m app.main
```

#### Run Specific Sections:
```bash
# 1. Standard Library Demos only
.venv/bin/python -m app.main --section python

# 2. Machine Learning & Data Science Demos only
.venv/bin/python -m app.main --section ml

# 3. AI & Agent Framework Demos only
.venv/bin/python -m app.main --section ai
```

### Running Tests

Execute the unit test suite with `pytest`:

```bash
.venv/bin/pytest tests/test_demos.py
```

### Running with `agents-cli`

Start the local interactive playground server:

```bash
agents-cli playground
```

---

## 📄 PDF Upload & Pinecone RAG API

This project includes a complete PDF ingestion, embedding, and semantic search RAG pipeline.

### Features
1. **PDF Parsing**: Extracts page-by-page text chunks and embedded images using `pypdf` and `Pillow`.
2. **Separate Vector Embeddings**: Text chunks and extracted images are embedded separately into 384-dimensional vector representations.
3. **Pinecone Vector Store**: Vectors are upserted into Pinecone with rich metadata (`content_type`, `page_number`, `filename`). Falls back to an in-memory vector store for offline/local execution.
4. **RAG Semantic Search & LLM**: Performs similarity search across Pinecone, builds structured context, and queries the LLM to generate an answer.

### API Endpoints

#### 1. Upload PDF (`POST /api/v1/pdf/upload`)
Uploads a PDF file, parses text and images, embeds them separately, and indexes vectors into Pinecone.

```bash
curl -X POST -F "file=@sample_document.pdf" http://localhost:8000/api/v1/pdf/upload
```

#### 2. Semantic Query & RAG (`POST /api/v1/pdf/query`)
Queries Pinecone for top matching text and image embeddings and generates an LLM response.

```bash
curl -X POST http://localhost:8000/api/v1/pdf/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the key information in the uploaded PDF", "top_k": 3}'
```

#### 3. Vector Stats (`GET /api/v1/pdf/stats`)
Returns current vector storage stats and Pinecone connection status.

```bash
curl http://localhost:8000/api/v1/pdf/stats
```

---

## 📚 Complete Function Reference

Below is a detailed description of every module and function included in this repository.

### 1. Python Standard Library Demos (`app/python_features_demo.py`)

File: [app/python_features_demo.py](file:///config/.gemini/antigravity/scratch/python-demo-agent/app/python_features_demo.py)

| Function / Class | Parameters | Description |
|------------------|------------|-------------|
| `TaskItem` | `id: int, name: str, tags: List[str], metadata: Dict[str, Any]` | A `@dataclass` representing a task item with custom type annotations and default field factories. |
| `TaskItem.to_dict()` | `self` | Converts the `TaskItem` dataclass instance into a plain dictionary representation. |
| `cached_fibonacci(n: int)` | `n: int` | Calculates the $N$-th Fibonacci number using `functools.lru_cache` for memoization. |
| `multiply(a: int, b: int)` | `a: int, b: int` | Helper multiplication function used to demonstrate function partials. |
| `double(b: int)` | `b: int` | A partial function created via `functools.partial(multiply, 2)` that doubles its input. |
| `demo_collections_and_itertools()` | None | Demonstrates `collections` (`Counter`, `defaultdict`, `deque`, `namedtuple`) and `itertools` (`chain`, `combinations`, `islice`). |
| `demo_pathlib_and_json(temp_dir: Path)` | `temp_dir: Path` | Demonstrates file directory creation, JSON serialization (`json.dumps`), writing to disk, and reading/parsing JSON (`json.loads`) using `pathlib.Path`. |
| `async_worker(worker_id: int, delay: float)` | `worker_id: int, delay: float` | Asynchronous worker function simulating a non-blocking delay using `asyncio.sleep`. |
| `demo_asyncio()` | None | Executes multiple `async_worker` tasks concurrently using `asyncio.gather`. |
| `sync_work_item(x: int)` | `x: int` | Synchronous helper function returning the square of an integer `x`. |
| `demo_concurrent_futures()` | None | Executes multi-threaded worker jobs using `concurrent.futures.ThreadPoolExecutor`. |
| `run_all_python_features_demo(output_dir: Optional[Path])` | `output_dir: Optional[Path]` | Orchestrates and runs all standard library feature demonstrations, returning a consolidated dictionary summary. |

---

### 2. Machine Learning & Data Science Demos (`app/ml_libraries_demo.py`)

File: [app/ml_libraries_demo.py](file:///config/.gemini/antigravity/scratch/python-demo-agent/app/ml_libraries_demo.py)

| Function / Class | Parameters | Description |
|------------------|------------|-------------|
| `demo_numpy()` | None | Demonstrates NumPy multi-dimensional array creation, reshaping, mean calculation (`np.mean`), and matrix vector dot product (`np.dot`). |
| `demo_pandas()` | None | Demonstrates Pandas `DataFrame` creation, row filtering based on conditions, and column mean calculation. |
| `demo_matplotlib(output_dir: Path)` | `output_dir: Path` | Generates a sine wave line chart in non-interactive headless mode and saves the image to `output_dir/sample_plot.png`. |
| `demo_scikit_learn()` | None | Creates a synthetic 1D dataset, splits train/test data using `train_test_split`, trains a `LinearRegression` model, and computes predictions. |
| `demo_pytorch()` | None | Defines a simple PyTorch `nn.Module` linear network layer and performs a forward pass using `torch.tensor` inputs. |
| `run_all_ml_libraries_demo(output_dir: Optional[Path])` | `output_dir: Optional[Path]` | Runs all Machine Learning & Data Science demonstrations and returns a structured summary. |

---

### 3. AI & Agent Framework Demos (`app/ai_agent_libraries_demo.py`)

File: [app/ai_agent_libraries_demo.py](file:///config/.gemini/antigravity/scratch/python-demo-agent/app/ai_agent_libraries_demo.py)

| Function / Class | Parameters | Description |
|------------------|------------|-------------|
| `demo_langchain()` | None | Builds a LangChain `PromptTemplate` and constructs an LCEL pipeline chain using `RunnableLambda`. |
| `step_1(state: AgentState)` | `state: AgentState` | First node transform function in the LangGraph workflow. |
| `step_2(state: AgentState)` | `state: AgentState` | Second node transform function in the LangGraph workflow. |
| `demo_langgraph()` | None | Constructs and compiles a stateful multi-step execution graph using `StateGraph`, `START`, and `END` nodes. |
| `demo_llama_index()` | None | Indexes in-memory `Document` objects into a `VectorStoreIndex` using `MockEmbedding` and `MockLLM`. |
| `demo_chromadb()` | None | Instantiates an ephemeral `chromadb.Client()`, creates/retrieves a collection, adds text documents with metadata, and queries vector similarity. |
| `demo_pinecone()` | None | Demonstrates Pinecone SDK initialization and client status checking. |
| `demo_sentence_transformers()` | None | Loads the `SentenceTransformer("all-MiniLM-L6-v2")` model and calculates dense vector embeddings for input sentences. |
| `demo_peft()` | None | Configures Parameter-Efficient Fine-Tuning parameters via `LoraConfig` (rank $r$, alpha, target modules). |
| `run_all_ai_agent_libraries_demo()` | None | Orchestrates and runs all AI & LLM Agent framework demonstrations. |

---

### 4. ADK Agent Tools (`app/tools.py`)

File: [app/tools.py](file:///config/.gemini/antigravity/scratch/python-demo-agent/app/tools.py)

| Function | Description |
|----------|-------------|
| `demo_python_features_tool()` | ADK Tool wrapper that executes and returns standard Python feature demonstrations. |
| `demo_ml_libraries_tool()` | ADK Tool wrapper that executes and returns Machine Learning & Data Science demonstrations. |
| `demo_ai_libraries_tool()` | ADK Tool wrapper that executes and returns AI Framework demonstrations. |

---

### 5. Main CLI Entrypoint (`app/main.py`)

File: [app/main.py](file:///config/.gemini/antigravity/scratch/python-demo-agent/app/main.py)

| Function | Parameters | Description |
|----------|------------|-------------|
| `main()` | CLI flags `--section`, `--output-dir` | Parses CLI arguments, runs the selected demo modules, and prints formatted JSON summary output. |

---

## 🛠️ Project Structure

```text
python-demo-agent/
├── app/
│   ├── __init__.py
│   ├── agent.py                      # ADK Agent definition & tool registration
│   ├── ai_agent_libraries_demo.py    # LangChain, LangGraph, LlamaIndex, Chroma, Pinecone, ST, PEFT
│   ├── fast_api_app.py               # FastAPI server backend
│   ├── main.py                       # CLI execution entrypoint
│   ├── ml_libraries_demo.py          # PyTorch, Scikit-Learn, NumPy, Matplotlib, Pandas
│   ├── python_features_demo.py      # collections, itertools, functools, typing, asyncio, etc.
│   └── tools.py                      # ADK Agent function tools
├── demo_output/                      # Saved charts & JSON files from demos
├── tests/
│   └── test_demos.py                 # Pytest unit tests suite
├── GEMINI.md                         # AGY project context guide
└── pyproject.toml                    # Package dependencies & configuration
```
