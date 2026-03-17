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
import re
from playwright.sync_api import Page
from google import genai
from google.genai import types

MAX_AGENT_STEPS = 20
MAX_PAGE_TEXT_LENGTH = 1000
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"

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


class _AdapterResponse:
    """Simple response wrapper so run_task can keep using response.text."""

    def __init__(self, text: str):
        self.text = text


class GeminiModelAdapter:
    """Compatibility layer that exposes generate_content like the old SDK."""

    def __init__(self, client: genai.Client, model_name: str):
        self._client = client
        self._model_name = model_name

    def generate_content(self, messages: list[dict], generation_config: dict | None = None):
        response_mime_type = "application/json"
        if generation_config and generation_config.get("response_mime_type"):
            response_mime_type = generation_config["response_mime_type"]

        prompt = _messages_to_prompt(messages)
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type=response_mime_type,
                system_instruction=_SYSTEM_PROMPT,
            ),
        )
        return _AdapterResponse((response.text or "").strip())


def _messages_to_prompt(messages: list[dict]) -> str:
    """Flatten role/parts messages into a prompt string for google.genai."""
    chunks = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        parts = message.get("parts", [])
        text = "\n".join(str(part) for part in parts)
        chunks.append(f"{role}:\n{text}")
    return "\n\n".join(chunks)


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
        try:
            response = model.generate_content(
                messages,
                generation_config={"response_mime_type": "application/json"},
            )
        except Exception as exc:
            return _run_fallback_task(page, task, str(exc))
        raw = response.text.strip()

        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            return f"Agent returned invalid JSON: {raw}"

        action_type = action.get("action")
        print(f"[Agent] Action: {action_type}")

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


def _run_fallback_task(page: Page, task: str, reason: str) -> str:
    """Run a tiny deterministic fallback flow when Gemini is unavailable."""
    lowered = task.lower()
    url_match = re.search(r"https?://\S+", task)

    if url_match:
        url = url_match.group(0).rstrip(".,)")
        print(f"[Agent] Fallback action: navigate -> {url}")
        page.goto(url, wait_until="domcontentloaded")

    if "title" in lowered:
        title = page.title()
        return (
            "Gemini unavailable (quota/API issue), fallback executed. "
            f"Page title: {title}."
        )

    return (
        "Gemini unavailable (quota/API issue), fallback executed basic navigation. "
        f"Reason: {reason}"
    )


def create_model(api_key: str | None = None) -> GeminiModelAdapter:
    """Configure and return a Gemini model adapter based on google.genai."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    if not key:
        raise ValueError(
            "A Gemini API key is required. "
            "Set the GEMINI_API_KEY environment variable or pass api_key='<your-key>'."
        )
    client = genai.Client(api_key=key)
    return GeminiModelAdapter(client=client, model_name=model_name)
