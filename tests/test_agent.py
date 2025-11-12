import pytest
from langsmith import testing as t
from src.workflow.agent_workflow import ReActAgent

@pytest.mark.langsmith
@pytest.mark.parametrize(
  # <-- Can still use all normal pytest markers
  "query",
  ["Hello!", "How are you doing?"],
)
def test_no_tools_on_offtopic_query(query: str) -> None:
  """Test that the agent does not use tools on offtopic queries."""

  agent = ReActAgent()
  tool_agent = agent.model_with_tools
  # Log the test example
  t.log_inputs({"query": query})
  expected = []
  t.log_reference_outputs({"tool_calls": expected})
  # Call the agent's model node directly instead of running the ReACT loop.
  result = tool_agent.nodes["agent"].invoke(
      {"messages": [{"role": "user", "content": query}]}
  )
  actual = result["messages"][0].tool_calls
  t.log_outputs({"tool_calls": actual})
  # Check that no tool calls were made.
  assert actual == expected

@pytest.mark.langsmith
def test_searches_for_correct_ticker() -> None:
  """Test that the model looks up the correct ticker on simple query."""
  agent = ReActAgent()
  tool_agent = agent.model_with_tools
  # Log the test example
  query = "What is the current weather in Hyderabad?"
  t.log_inputs({"query": query})
  expected = "Hyderabad"
  t.log_reference_outputs({"ticker": expected})
  # Call the agent's model node directly instead of running the full ReACT loop.
  result = tool_agent.nodes["agent"].invoke(
      {"messages": [{"role": "user", "content": query}]}
  )
  tool_calls = result["messages"][0].tool_calls
  if tool_calls[0]["name"] == get_weather_data.name:
      actual = tool_calls[0]["args"]["ticker"]
  else:
      actual = None
  t.log_outputs({"ticker": actual})
  # Check that the right ticker was queried
  assert actual == expected