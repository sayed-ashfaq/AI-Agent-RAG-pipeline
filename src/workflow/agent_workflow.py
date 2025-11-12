from typing import List, Literal
from typing_extensions import TypedDict, Annotated
import operator

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.tools import tool
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from src.prompt_library.system_prompt import AGENT_SYSTEM_PROMPT_CONCISE
from src.tools.rag_agent.document_loader import DocumentService
from src.tools.rag_agent.retriever import Retriever
from src.utils.model_loader import ModelLoader
from custom_logger import GLOBAL_LOGGER as logger
from exception_handler.agent_exceptions import WorkflowError, FileProcessingError, RetrieverError


class AgentRAG:
    class AgentState(TypedDict):
        messages: Annotated[List[AnyMessage], operator.add]
        llm_calls: int

    def __init__(self, retriever= None):
        self.retriever = retriever if retriever is not None else Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.system_prompt = AGENT_SYSTEM_PROMPT_CONCISE

        self.checkpointer = MemorySaver()

        self.model_with_tools = self._load_tools()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    # ------- Helpers -------
    def document_loader(self, path):
        try:
            loader = DocumentService()
            return loader.process_single_file(file_path=path, file_type="pdf")
        except Exception as e:
            logger.error(f"Unable to load document: {path} | Error: {e}")
            raise FileProcessingError(f"Unable to load document: {path}")

    def _format_docs(self, docs) -> str:
        if not docs:
            return "No relevant documents found."
        return "\n\n---\n\n".join([doc.page_content.strip() for doc in docs])

    def _load_tools(self):
        try:
            @tool
            def get_document_context(query: str):
                """RAG that gets the context based on query"""
                try:
                    docs = self.retriever.call_retriever(query)
                    context = self._format_docs(docs)
                    logger.info("Successfully initiated retriever tool")
                    return context
                except Exception as e:
                    logger.error(f"Failed to initiate retriever tool: {e}")
                    raise RetrieverError("Failed to initiate retriever tool", str(e))

            @tool
            def get_weather_data(location: str):
                """Fetches weather data from OpenWeatherMap based on location"""
                try:
                    weather = OpenWeatherMapAPIWrapper()
                    logger.info("Initiated weather tool via OpenWeatherMap")
                    return weather.run(location)
                except Exception as e:
                    logger.error(f"Unable to fetch weather data: {e}")
                    raise WorkflowError(f"Unable to fetch weather data: {e}")

            self.tools = [get_document_context, get_weather_data]
            self.tools_by_name = {t.name: t for t in self.tools}

            return self.llm.bind_tools(self.tools)

        except Exception as e:
            logger.error(f"Failed to load tools to the LLM: {e}")
            raise WorkflowError("Failed to load tools to the LLM")

    # ------- Node functions -------
    def _llm_node(self, state: AgentState):
        """LLM node: decides whether to call a tool"""
        messages = state["messages"]

        result = self.model_with_tools.invoke(
            [SystemMessage(content=self.system_prompt)]
            + messages
        )

        # KEY FIX: Only return the new message, not messages + [result]
        # Because operator.add will append it automatically
        return {
            "messages": [result],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def _tool_node(self, state: AgentState):
        """Handles tool invocations"""
        messages = state["messages"]
        last_message = messages[-1]
        results = []

        for tool_call in getattr(last_message, "tool_calls", []):
            try:
                tool = self.tools_by_name[tool_call["name"]]
                # Ensure args are unpacked correctly
                tool_args = tool_call.get("args", {})
                context = tool.invoke(tool_args)
                results.append(
                    ToolMessage(content=context, tool_call_id=tool_call["id"])
                )
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                results.append(ToolMessage(content=f"Tool execution failed: {e}", tool_call_id=tool_call["id"]))

        # KEY FIX: Only return the new tool messages, not messages + results
        return {"messages": results}

    def should_continue(self, state: AgentState) -> Literal["tool_node", END]:
        """Determine if workflow should continue"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_node"
        return END

    # ------- Workflow -------
    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)

        workflow.add_node("agent", self._llm_node)
        workflow.add_node("tool_node", self._tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", self.should_continue, ["tool_node", END])
        workflow.add_edge("tool_node", "agent")

        return workflow

    # ------- Runner -------
    def run(self, query: str, thread_id: str = "default_thread") -> str:
        """Run the workflow for a given query and return the final LLM answer"""
        output = self.app.invoke(
            {"messages": [HumanMessage(content=query)], "llm_calls": 0},
            config={"configurable": {"thread_id": thread_id}},
        )

        final_msg = output["messages"][-1]
        return getattr(final_msg, "content", "No final content returned.")

if __name__ == "__main__":
    agent = AgentRAG()
    print("="*25,"First Question","="*25)
    print(agent.run(query="What is your name?"))
    print("="*25,"Second Question","="*25)
    print(agent.run(query= "Can i go out in hyderabad, what's the weather today"))
    print("="*25,"Third Question","="*25)