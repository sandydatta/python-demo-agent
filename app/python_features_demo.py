"""
Demonstration of core Python standard library features:
- collections
- itertools
- functools
- typing
- dataclasses
- asyncio
- concurrent.futures
- pathlib
- logging
- json
"""

import asyncio
import concurrent.futures
from collections import Counter, defaultdict, deque, namedtuple
from dataclasses import dataclass, field
import functools
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# 1. dataclasses & typing
@dataclass
class TaskItem:
    id: int
    name: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tags": self.tags,
            "metadata": self.metadata,
        }


# 2. functools - caching & partials
@functools.lru_cache(maxsize=32)
def cached_fibonacci(n: int) -> int:
    if n < 2:
        return n
    return cached_fibonacci(n - 1) + cached_fibonacci(n - 2)


def multiply(a: int, b: int) -> int:
    return a * b


double = functools.partial(multiply, 2)


# 3. collections & itertools
def demo_collections_and_itertools() -> Dict[str, Any]:
    logger.info("Running collections & itertools demo...")
    
    # Collections
    counts = Counter(["apple", "banana", "apple", "orange", "banana", "apple"])
    default_map = defaultdict(list)
    default_map["fruits"].append("apple")
    default_map["fruits"].append("banana")
    
    q = deque([1, 2, 3])
    q.appendleft(0)
    q.append(4)
    
    Point = namedtuple("Point", ["x", "y"])
    p = Point(10, 20)

    # Itertools
    chained = list(itertools.chain([1, 2], [3, 4]))
    combinations = list(itertools.combinations(["A", "B", "C"], 2))
    sliced = list(itertools.islice(range(100), 5))

    return {
        "most_common_fruit": counts.most_common(1)[0],
        "defaultdict": dict(default_map),
        "deque": list(q),
        "point": {"x": p.x, "y": p.y},
        "chained": chained,
        "combinations": combinations,
        "sliced": sliced,
    }


# 4. pathlib & json
def demo_pathlib_and_json(temp_dir: Path) -> Dict[str, Any]:
    logger.info("Running pathlib & json demo...")
    temp_dir.mkdir(parents=True, exist_ok=True)
    json_path = temp_dir / "sample_data.json"

    data = {
        "project": "python-demo-agent",
        "features": ["collections", "dataclasses", "asyncio", "pathlib", "json"],
        "status": "success",
    }

    # Pathlib write & json dump
    json_path.write_text(json.dumps(data, indent=2))
    
    # Pathlib read & json load
    read_data = json.loads(json_path.read_text())
    
    return {
        "file_path": str(json_path),
        "file_exists": json_path.exists(),
        "data_read": read_data,
    }


# 5. asyncio
async def async_worker(worker_id: int, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"Worker {worker_id} completed in {delay}s"


async def demo_asyncio() -> List[str]:
    logger.info("Running asyncio demo...")
    tasks = [
        async_worker(1, 0.1),
        async_worker(2, 0.2),
        async_worker(3, 0.05),
    ]
    results = await asyncio.gather(*tasks)
    return list(results)


# 6. concurrent.futures
def sync_work_item(x: int) -> int:
    return x * x


def demo_concurrent_futures() -> List[int]:
    logger.info("Running concurrent.futures demo...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(sync_work_item, i) for i in range(5)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    return sorted(results)


def run_all_python_features_demo(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = Path("./demo_output")
    
    task = TaskItem(id=101, name="Standard Library Demo", tags=["python", "stdlib"])
    fib_res = cached_fibonacci(10)
    doubled_res = double(21)
    coll_iter_res = demo_collections_and_itertools()
    path_json_res = demo_pathlib_and_json(output_dir)
    async_res = asyncio.run(demo_asyncio())
    futures_res = demo_concurrent_futures()

    summary = {
        "dataclass_task": task.to_dict(),
        "fibonacci_cached(10)": fib_res,
        "partial_double(21)": doubled_res,
        "collections_and_itertools": coll_iter_res,
        "pathlib_and_json": path_json_res,
        "asyncio_results": async_res,
        "concurrent_futures_results": futures_res,
    }
    logger.info("Python Standard Library Demos completed successfully.")
    return summary


if __name__ == "__main__":
    res = run_all_python_features_demo()
    print(json.dumps(res, indent=2))
