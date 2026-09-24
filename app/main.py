"""
Main local entrypoint for python-demo-agent.
Runs all Python features, ML libraries, and AI/Agent framework demos locally.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

from app.ai_agent_libraries_demo import run_all_ai_agent_libraries_demo
from app.ml_libraries_demo import run_all_ml_libraries_demo
from app.python_features_demo import run_all_python_features_demo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("python-demo-agent")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Python Features & AI Libraries Demo")
    parser.add_argument(
        "--section",
        choices=["all", "python", "ml", "ai"],
        default="all",
        help="Section to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./demo_output",
        help="Output directory for generated files/plots",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if args.section in ["all", "python"]:
        logger.info("=== 1. Python Standard Library Demos ===")
        results["python_stdlib"] = run_all_python_features_demo(out_dir)

    if args.section in ["all", "ml"]:
        logger.info("=== 2. ML & Data Science Libraries Demos ===")
        results["ml_libraries"] = run_all_ml_libraries_demo(out_dir)

    if args.section in ["all", "ai"]:
        logger.info("=== 3. AI & Agent Framework Demos ===")
        results["ai_agent_libraries"] = run_all_ai_agent_libraries_demo()

    print("\n" + "=" * 60)
    print("           PYTHON DEMO AGENT - TEST & RUN SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
