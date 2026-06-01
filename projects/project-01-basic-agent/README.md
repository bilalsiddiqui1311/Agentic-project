# Project 01: Basic Tool-Using Agent

Goal: learn the smallest useful agent loop.

```text
user message -> choose tool -> run tool -> observe result -> final answer
```

What this project teaches:

- An agent is a controller around tools.
- Tools are ordinary functions with clear inputs and outputs.
- The response should show the decision steps.
- Docker Compose should run the API the same way for everyone.

Main files:

- `app/agents/basic_agent.py`
- `app/tools/calculator.py`
- `app/tools/knowledge.py`
- `app/main.py`
- `tests/test_basic_agent.py`

Branch:

- `codex/project-01-basic-agent`

Tag:

- `project-01`
