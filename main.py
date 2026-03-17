"""
Entry point for the Playwright + Gemini agentic browser demo.

Usage:
    python main.py [--tasks tasks.json] [--headless]

Environment variables:
    GEMINI_API_KEY  – required Gemini API key

The script reads a list of task descriptions from a JSON file, then runs each
task in a Playwright browser session, printing the agent's result.
"""

import argparse
import json
import sys

from playwright.sync_api import sync_playwright

from agent import create_model, run_task


def load_tasks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path!r}, got {type(data).__name__}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic browser demo with Playwright + Gemini")
    parser.add_argument(
        "--tasks",
        default="tasks.json",
        help="Path to a JSON file containing the list of tasks (default: tasks.json)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run the browser in headless mode (default: False)",
    )
    args = parser.parse_args()

    try:
        tasks = load_tasks(args.tasks)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading tasks: {exc}", file=sys.stderr)
        return 1

    try:
        model = create_model()
    except ValueError as exc:
        print(f"Error creating model: {exc}", file=sys.stderr)
        return 1

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless)
        page = browser.new_page()

        for i, task_obj in enumerate(tasks, start=1):
            description = task_obj.get("description", "").strip()
            if not description:
                print(f"[Task {i}] Skipped – missing 'description' field.")
                continue

            print(f"\n[Task {i}] {description}")
            result = run_task(page, description, model)
            print(f"[Task {i}] Result: {result}")

        browser.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
