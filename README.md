# Playwright Demo – Agentic Browser Automation

A minimal demo that combines **Playwright** (browser automation) with **Google Gemini** (AI) to perform agentic browser tasks described in plain English.

## How it works

1. You define a list of tasks in `tasks.json` (natural-language descriptions).
2. `main.py` feeds each task to the **Gemini** model together with the current page state.
3. Gemini returns a JSON action (`navigate`, `click`, `type`, `get_text`, or `done`).
4. Playwright executes the action in a real browser.
5. The updated page state is sent back to Gemini, repeating until the model returns `done`.

## Project structure

```
playwright_demo/
├── agent.py          # Playwright + Gemini agent loop
├── main.py           # CLI entry point
├── tasks.json        # Input: list of tasks to run
├── requirements.txt  # Python dependencies
└── README.md
```

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Playwright browsers
python -m playwright install chromium

# 3a. Set your Gemini API key (PowerShell)
$env:GEMINI_API_KEY="your-api-key-here"

# 3b. OR create a .env file with:
# GEMINI_API_KEY=your-api-key-here
```

## Usage

```bash
# Run with the default tasks.json (headed browser)
python main.py

# Run headless
python main.py --headless

# Use a custom tasks file
python main.py --tasks my_tasks.json
```

## tasks.json format

```json
[
  { "description": "Go to https://example.com and get the page title" },
  { "description": "Search for 'Playwright' on https://playwright.dev" }
]
```
