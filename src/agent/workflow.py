import enum
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from typing_extensions import Annotated, TypedDict, Literal

from src.agent import prompt, tool as agent_tool, utils as agent_utils
from src.tools.startup import settings

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledGraph, CompiledStateGraph

if settings.get("MODEL_NAME") == agent_utils.ModelName.openai.value:
    _model_name = agent_utils.ModelName.openai.value
else:
    _model_name = agent_utils.ModelName.google.value


class GraphState(TypedDict):
    """
    Represents the state of the multi-agent graph system.

    Attributes:
        question (str): The original query from the human user.
        messages (list): Tracked messages with automatic addition
            functionality (via Annotated).
        history_messages (str): String representation of conversation
            history for context.
        next_agent (str): Identifier for the next agent to execute.
        next_input (str): Input text to provide to the next agent.
        draft_answer (str): Initial response generated before verification.
        answer_feedback (str): Feedback on the draft answer from verifier.
        final_answer (str): The verified and refined response to return.
        image (str): Base64-encoded image data if applicable to the query.
    """
    question: str
    messages: Annotated[list, add_messages]
    history_messages: str
    next_agent: str
    next_input: str
    draft_answer: str
    answer_feedback: str
    final_answer: str
    image: str


class AgentName(enum.Enum):
    """
    Enumeration of agent types available in the system.

    Each agent has a specific role in the collaborative reasoning pipeline:
    - router: determines which agent should handle the query initially
    - researcher: gathers information from external sources
    - reasoner: analyzes information and formulates logical conclusions
    - data_analyst: analyzes the data source like excel, csc files
    - generator: produces draft and final answers
    - verifier: validates the generated answers for accuracy and completeness
    """
    router = "router"
    researcher = "researcher"
    reasoner = "reasoner"
    data_analyst = "data_analyst"
    generator = "generator"
    verifier = "verifier"


class QuestionAnsweringGraph:
    """
    Multi-agent question answering system using LangGraph.

    This class orchestrates a collaborative system of specialized agents
    that work together to answer complex queries. The system includes routing,
    research, reasoning, data analysis, answer generation, and verification
    capabilities.

    The workflow follows a structured pipeline:
    1. Router determines the appropriate expert agent
    2. Researcher gathers external information (if needed)
    3. Reasoner or Data Analyst processes information
    4. Generator creates draft answers
    5. Verifier validates and provides feedback
    6. Generator produces final refined answers

    Attributes:
        _router_name (str): Name identifier for the router agent.
        _reasoner_name (str): Name identifier for the reasoner agent.
        _researcher_name (str): Name identifier for the researcher agent.
        _data_analyst_name (str): Name identifier for the data analyst agent.
        _generator_name (str): Name identifier for the generator agent.
        _verifier_name (str): Name identifier for the verifier agent.
        _agents_to_route (list): List of available agents for
            routing decisions.
    """
    def __init__(self) -> None:
        """
        Initialize the QuestionAnsweringGraph with agent configurations.

        Sets up the agent names and defines the available agents that can be
        routed to by the router agent. Currently, supports routing to reasoner,
        researcher, and data analyst agents.
        """
        # Agents names
        self._router_name = AgentName.router.name
        self._reasoner_name = AgentName.reasoner.name
        self._researcher_name = AgentName.researcher.name
        self._data_analyst_name = AgentName.data_analyst.name
        self._generator_name = AgentName.generator.name
        self._verifier_name = AgentName.verifier.name

        # The available agents to route are the following
        # - Reasoner
        # - Researcher
        # - Data analyst
        self._agents_to_route = [
            self._reasoner_name,
            self._reasoner_name,
            self._researcher_name,
        ]

    @staticmethod
    def _invoke(agent: "CompiledGraph", state: GraphState) -> dict[str, Any]:
        """
        Invoke an agent with the given state and return the response.

        This is a utility method that provides a consistent interface for
        invoking any agent in the system with the current graph state.

        Args:
           agent (CompiledGraph): The compiled agent graph to invoke.
           state (GraphState): The current state containing all context
               and conversation history.

        Returns:
           dict[str, Any]: The response from the agent invocation, typically
               containing messages, structured responses, and updated state.
        """
        return agent.invoke(state)

    def _router_node(self, state: GraphState) \
            -> Command[Literal["researcher", "reasoner", "data_analyst"]]:
        """
        Determines which specialized agent should handle the query.

        This node acts as the entry point for the agent workflow. It analyzes.
        the query and routes it to either the researcher (for information
        gathering) or the reasoner (for analytical processing) based on the
        query characteristics.

        Args:
            state (GraphState): The current state of the agent graph.

        Returns:
            Command: A routing command directing flow to either the researcher
                or reasoner.
        """
        # Create a validation model for the router's output.
        router = agent_utils.create_router(agents=self._agents_to_route)
        router_agent = create_react_agent(
            model=agent_utils.get_model(
                model_name=_model_name,
                temperature=settings["TEMPERATURE"]),
            tools=[],
            prompt=prompt.ROUTER,
            name=self._router_name,
            response_format=router,
        )
        response = self._invoke(router_agent, state)

        # Format message for conversation history
        message_str = f"Route to " \
                      f"{response['structured_response'].expert_agent} " \
                      f"agent with input " \
                      f"{response['structured_response'].agent_input}"

        # Adding question
        history_messages = f"**Human** query: {state['question']}\n\n"
        history_messages += \
            (f"**{self._router_name.capitalize()} agent**: "
             f"{message_str}\n\n")

        return Command(
            goto=response["structured_response"].expert_agent,
            update={
                "messages": [
                    AIMessage(
                        content=message_str,
                        name=self._router_name,
                    )
                ],
                "history_messages": history_messages,
                "next_agent": response["structured_response"].expert_agent,
                "next_input": response["structured_response"].agent_input
            },
        )

    def _reasoner_node(
            self, state: GraphState) -> Command[Literal["generator"]]:
        """
        Analyzes information and develops logical conclusions.

        The reasoner agent examines the query and any available information
        to form reasoned conclusions. It can work directly with the query or
        with research provided by the researcher agent. After reasoning,
        it passes control to the generator agent.

        Args:
            state (GraphState): The current state of the agent graph.

        Returns:
            Command: A directive to proceed to the generator node with
                reasoning results.
        """
        reasoner_agent = create_react_agent(
            model=agent_utils.get_model(
                model_name=_model_name,
                temperature=settings["TEMPERATURE"]),
            tools=[],
            prompt=prompt.REASONING.format(
                question=state["question"],
                history_messages=state["history_messages"]),
            name=self._reasoner_name
        )

        # Prepare the human message
        human_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": state["next_input"]},
            ]
        }
        # If there is an image, add it to messages
        if base64_image := state["image"]:
            human_message["content"].append({
                "type": "image_url", "image_url": {"url": base64_image}})

        # Add the human message to the conversation
        state["messages"].append(human_message)

        response = self._invoke(reasoner_agent, state)
        last_message = agent_utils.get_last_message(response)

        # Update conversation history with reasoner's contribution
        history_messages = agent_utils.get_updated_history_messages(
            last_message, state["history_messages"], self._reasoner_name)

        return Command(
            goto=self._generator_name,
            update={
                "messages": [
                    AIMessage(
                        content=last_message,
                        name=self._reasoner_name,
                    )
                ],
                "history_messages": history_messages,
                "next_agent": self._generator_name,
                "next_input": last_message
            }
        )

    def _data_analyst_node(
            self, state: GraphState) -> Command[Literal["generator"]]:
        """
        Performs data analysis on provided information using Python execution
        capabilities.

        This node represents a specialized agent that can run Python code to
        analyze data within the multi-agent system. It uses the PythonReplTool
        to execute code and analyze data relevant to the user's query. After
        performing analysis, it passes control to the generator agent which can
        incorporate the analytical findings into the final response.

        Args:
            state (GraphState): The current state of the agent graph.

        Returns:
            Command: A directive to proceed to the generator node with the
                results of the data analysis.

        Note:
            The agent uses the PythonReplTool which executes Python code in a
            sandboxed environment. This enables data manipulation, calculation,
            and visualization capabilities.

        ** Take care with the security, the agent can execute any
        generated code **
        """
        data_analyst_agent = create_react_agent(
            model=agent_utils.get_model(
                model_name=_model_name,
                temperature=settings["TEMPERATURE"]),
            tools=[agent_tool.PythonReplTool()],
            prompt=prompt.DATA_ANALYST.format(
                question=state["question"],
                history_messages=state["history_messages"]),
            name=self._data_analyst_name
        )

        # Prepare the human message
        human_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": state["next_input"]},
            ]
        }

        # Add the human message to the conversation
        state["messages"].append(human_message)

        response = self._invoke(data_analyst_agent, state)
        last_message = agent_utils.get_last_message(response)

        # Update conversation history with data analyst's contribution
        history_messages = agent_utils.get_updated_history_messages(
            last_message, state["history_messages"], self._data_analyst_name)

        return Command(
            goto=self._generator_name,
            update={
                "messages": [
                    AIMessage(
                        content=last_message,
                        name=self._data_analyst_name,
                    )
                ],
                "history_messages": history_messages,
                "next_agent": self._generator_name,
                "next_input": last_message
            }
        )

    def _researcher_node(
            self, state: GraphState) -> Command[Literal["reasoner"]]:
        """
        Gathers information from external sources to answer the query.

        The researcher uses various tools (search, academic papers, wiki) to
        collect relevant information about the query. After gathering
        information, it passes control to the reasoner to analyze the findings.

        Args:
            state (GraphState): The current state of the agent graph.

        Returns:
            Command: A directive to proceed to the reasoner node with
                research results.
        """
        research_agent = create_react_agent(
            model=agent_utils.get_model(
                model_name=_model_name,
                temperature=settings["TEMPERATURE"]),
            tools=[
                agent_tool.WikipediaTool(),
                agent_tool.ArxivTool(),
                agent_tool.TavilySearchTool(),
                agent_tool.GetYoutubeUrlTranscription(),
                agent_tool.YoutubeVideoQuery()
            ],
            prompt=prompt.RESEARCH.format(
                history_messages=state["history_messages"]),
            name=self._researcher_name
        )

        # Add the original question to the conversation
        state["messages"].append(HumanMessage(state["question"]))
        response = self._invoke(research_agent, state)

        # Update conversation history with research findings
        last_message = agent_utils.get_last_message(response)
        history_messages = agent_utils.get_updated_history_messages(
            last_message, state["history_messages"], self._researcher_name)

        return Command(
            goto=self._reasoner_name,
            update={
                "messages": [
                    AIMessage(
                        content=last_message,
                        name=self._researcher_name,
                    )
                ],
                "history_messages": history_messages,
                "next_agent": self._reasoner_name,
                "next_input": last_message
            }
        )

    def _generator_node(
            self, state: GraphState) -> Command[Literal["verifier"]]:
        """
        Produces answers based on reasoning and optional feedback.

        The generator creates responses to the original query. It operates in
        two modes:
        1. Initial generation: Creates a draft answer that will be sent
            to verification
        2. Revised generation: Creates a final answer incorporating verifier
            feedback

        Args:
            state (GraphState): The current state of the agent graph.

        Returns:
            Command: A directive to either:
                     - proceed to the verifier (for draft answers)
                     - end the process (for final answers after feedback)
        """
        # Determine operating mode based on presence of feedback
        if state["answer_feedback"]:
            # Feedback exists - this is final answer generation
            next_agent = END
            generator_prompt = prompt.GENERATOR_WITH_FEEDBACK.format(
                draft_answer=state["draft_answer"],
                answer_feedback=state["answer_feedback"],
                history_messages=state["history_messages"])
        else:
            # No feedback - this is draft answer generation
            next_agent = self._verifier_name
            generator_prompt = prompt.GENERATOR_WITHOUT_FEEDBACK.format(
                history_messages=state["history_messages"])

        generator_agent = create_react_agent(
            model=agent_utils.get_model(
                model_name=_model_name,
                temperature=settings["TEMPERATURE"]),
            tools=[],
            prompt=generator_prompt,
            name=self._generator_name
        )

        # Add the original question to the conversation
        state["messages"].append(HumanMessage(state["question"]))
        response = self._invoke(generator_agent, state)

        # Update conversation history
        last_message = agent_utils.get_last_message(response)
        history_messages = agent_utils.get_updated_history_messages(
            last_message, state["history_messages"], self._generator_name)

        return Command(
            goto=next_agent,
            update={
                "messages": [
                    AIMessage(
                        content=last_message,
                        name=self._generator_name,
                    )
                ],
                "history_messages": history_messages,
                "next_agent": self._generator_name,
                "next_input": last_message,
                "answer_feedback": "",
                # When there is no answer feedback, store as draft response
                "draft_answer": last_message if not state[
                    "answer_feedback"] else "",
                # When there is answer feedback, store as final answer
                "final_answer": last_message if state[
                    "answer_feedback"] else "",
            }
        )

    def _verifier_node(
            self, state: GraphState) -> Command[Literal["generator"]]:
        """
        Evaluates the draft answer for accuracy, completeness, and quality.

        The verifier analyzes the draft answer produced by the generator
        and provides feedback for improvements. This feedback is then used by
        the generator to create a refined final answer.

        Args:
            state (GraphState): The current state of the agent graph.

        Returns:
            Command: A directive to return to the generator with verification
                feedback.
        """
        verifier_agent = create_react_agent(
            model=agent_utils.get_model(
                model_name=_model_name,
                temperature=settings["TEMPERATURE"]),
            tools=[],
            prompt=prompt.VERIFIER.format(
                answer=state["next_input"],
                history_messages=state["history_messages"]),
            name=self._verifier_name
        )
        # Add the original question to the conversation
        state["messages"].append(HumanMessage(state["question"]))
        response = self._invoke(verifier_agent, state)

        # Update conversation history with verification feedback
        last_message = agent_utils.get_last_message(response)
        history_messages = agent_utils.get_updated_history_messages(
            last_message, state["history_messages"], self._verifier_name)

        return Command(
            goto=self._generator_name,
            update={
                "messages": [
                    AIMessage(
                        content=last_message,
                        name=self._verifier_name,
                    )
                ],
                "history_messages": history_messages,
                "next_agent": self._generator_name,
                "next_input": last_message,
                # Adding the feedback for the previous generator answer
                "answer_feedback": last_message,
            }
        )

    def build(self) -> "CompiledStateGraph":
        """
        Constructs and returns a compiled multi-agent conversation graph.

        This function creates a directed graph that orchestrates the flow of
        information between specialized agent nodes. The graph starts with the
        router, which directs the query to either the researcher or reasoner
        agent. The flow then proceeds through reasoning, generation, and
        verification steps as appropriate.

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
        builder = StateGraph(GraphState)
        # Define the starting point, connecting to the router
        builder.add_edge(START, self._router_name)
        # Add all agent nodes with their respective handler functions
        builder.add_node(self._router_name, self._router_node)
        builder.add_node(self._reasoner_name, self._reasoner_node)
        builder.add_node(self._researcher_name, self._researcher_node)
        builder.add_node(self._data_analyst_name, self._data_analyst_node)
        builder.add_node(self._generator_name, self._generator_node)
        builder.add_node(self._verifier_name, self._verifier_node)

        memory = MemorySaver()
        return builder.compile(checkpointer=memory)
