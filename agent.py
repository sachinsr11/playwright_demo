"""
Minimal Playwright + Gemini agentic browser demo.

The agent reads a task description, asks Gemini to generate a step-by-step
action plan, then executes each action in a live browser via Playwright.

Supported Gemini actions (returned as JSON by the model):
  {"action": "navigate", "url": "<url>"}
  {"action": "click",    "selector": "<css-selector>"}
  {"action": "type",     "selector": "<css-selector>", "text": "<text>"}
  {"action": "get_text", "selector": "<css-selector>"}
  {"action": "done",     "result": "<final answer>"}
"""

import json
import os
from playwright.sync_api import Page
import google.generativeai as genai

MAX_AGENT_STEPS = 20
MAX_PAGE_TEXT_LENGTH = 1000

_SYSTEM_PROMPT = """You are a browser automation assistant.
Given a task description and the current page state (URL + visible text),
return ONE JSON action to perform next.

Available actions:
  {"action": "navigate", "url": "<url>"}
  {"action": "click",    "selector": "<css-selector>"}
  {"action": "type",     "selector": "<css-selector>", "text": "<text>"}
  {"action": "get_text", "selector": "<css-selector>"}
  {"action": "done",     "result": "<final answer / summary>"}

Rules:
- Reply with valid JSON only – no markdown, no explanation.
- Use "done" when the task is complete or you have the answer.
- Keep selectors simple and robust (e.g. "h1", "a[href*='download']").
"""


def _page_state(page: Page) -> str:
    """Return a compact representation of the current page."""
    url = page.url
    try:
        body_text = page.inner_text("body")[:MAX_PAGE_TEXT_LENGTH]
    except Exception:
        body_text = "(could not read page text)"
    return f"URL: {url}\nPage text (first 1000 chars):\n{body_text}"


def run_task(page: Page, task: str, model) -> str:
    """
    Run a single task using the Gemini model and Playwright.

    Args:
        page:  An open Playwright page.
        task:  Natural-language task description.
        model: A google.generativeai GenerativeModel instance.

    Returns:
        The final result / answer produced by the agent.
    """
    messages = [
        {"role": "user", "parts": [f"Task: {task}\n\n{_page_state(page)}"]},
    ]

    for _ in range(MAX_AGENT_STEPS):  # safety cap
        response = model.generate_content(
            messages,
            generation_config={"response_mime_type": "application/json"},
        )
        raw = response.text.strip()

        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            return f"Agent returned invalid JSON: {raw}"

        action_type = action.get("action")

        if action_type == "navigate":
            page.goto(action["url"], wait_until="domcontentloaded")
        elif action_type == "click":
            page.click(action["selector"])
            page.wait_for_load_state("domcontentloaded")
        elif action_type == "type":
            page.fill(action["selector"], action["text"])
        elif action_type == "get_text":
            text = page.inner_text(action["selector"])
            messages.append({"role": "model", "parts": [raw]})
            messages.append(
                {"role": "user", "parts": [f"Element text: {text}\n\n{_page_state(page)}"]}
            )
            continue
        elif action_type == "done":
            return action.get("result", "Task complete.")
        else:
            return f"Unknown action: {action_type}"

        # After mutating actions, feed updated page state back to the model
        messages.append({"role": "model", "parts": [raw]})
        messages.append({"role": "user", "parts": [_page_state(page)]})

    return "Max steps reached without completion."


def create_model(api_key: str | None = None) -> genai.GenerativeModel:
    """Configure and return a Gemini GenerativeModel."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "A Gemini API key is required. "
            "Set the GEMINI_API_KEY environment variable or pass api_key='<your-key>'."
        )
    genai.configure(api_key=key)
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=_SYSTEM_PROMPT,
    )
