from app.agents.basic_agent import build_basic_agent


def test_basic_agent_uses_knowledge_search_for_rag_question() -> None:
    agent = build_basic_agent()

    response = agent.run("What is RAG?")

    assert response.used_tool == "knowledge_search"
    assert "Retrieval-Augmented Generation" in response.final_answer
    assert [step.kind for step in response.steps] == ["thought", "action", "observation", "final"]


def test_basic_agent_uses_calculator_for_arithmetic() -> None:
    agent = build_basic_agent()

    response = agent.run("Calculate 42 * 7")

    assert response.used_tool == "calculator"
    assert "294" in response.final_answer


def test_basic_agent_returns_clean_calculator_errors() -> None:
    agent = build_basic_agent()

    response = agent.run("Calculate 10 / 0")

    assert response.used_tool == "calculator"
    assert "Division by zero is not allowed." in response.final_answer
