from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END

from src.agent import node, utils as agent_utils

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_graph():
    """
    Constructs and returns a compiled multi-agent conversation graph.

    This function creates a directed graph that orchestrates the flow of
    information between specialized agent nodes. The graph starts with the
    router, which directs the query to either the researcher or reasoner agent.
    The flow then proceeds through reasoning, generation, and verification
    steps as appropriate.

    Graph flow:
    - START -> router
    - router -> researcher or reasoner or data analyst (decided by router)
    - researcher -> reasoner
    - reasoner -> generator
    - data analyst -> generator
    - generator -> verifier (for draft answers)
    - verifier -> generator (for refinement)
    - generator -> END (for final answers)

    Returns:
        CompiledStateGraph: A compiled state graph that can be executed to
            process queries through the multi-agent system.
    """
    builder = StateGraph(node.GraphState)
    # Define the starting point, connecting to the router
    builder.add_edge(START, agent_utils.AgentName.router.name)
    # Add all agent nodes with their respective handler functions
    builder.add_node(agent_utils.AgentName.router.name, node.router_node)
    builder.add_node(agent_utils.AgentName.reasoner.name, node.reasoner_node)
    builder.add_node(
        agent_utils.AgentName.researcher.name, node.researcher_node)
    builder.add_node(
        agent_utils.AgentName.data_analyst.name, node.data_analyst_node)
    builder.add_node(agent_utils.AgentName.generator.name, node.generator_node)
    builder.add_node(agent_utils.AgentName.verifier.name, node.verifier_node)
    # Define the end point - the generator can finish the process when
    # producing final answer
    builder.add_edge("generator", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
