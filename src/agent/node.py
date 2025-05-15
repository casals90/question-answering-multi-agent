from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict, Literal

from src.agent import prompt, utils as agent_utils
from src.agent import tool
from src.tools.startup import settings

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


def create_router(options: list[str]) -> type[BaseModel]:
    """
    Creates a Pydantic model for validating routing decisions.

    This function dynamically generates a Pydantic model class that enforces
    validation rules for routing between agents. The expert_agent field
    can only contain values from the provided options list.

    Args:
        options (list[str]): List of valid agent names for routing.

    Returns:
        type[BaseModel]: A Pydantic model class with next_agent
            field validation.
    """

    class Router(BaseModel):
        expert_agent: str = Field(
            description="The expert agent to route", enum=options)
        agent_input: str = Field(
            description="The input of expert agent"
        )

    return Router


def router_node(state: GraphState) -> Literal["researcher", "reasoner"]:
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
    # Create a validation model for the router's output
    router = create_router(options=[
        agent_utils.AgentName.reasoner.name,
        agent_utils.AgentName.researcher.name
    ])
    router_agent = create_react_agent(
        model=agent_utils.get_model(model_name=_model_name),
        tools=[],
        prompt=prompt.ROUTER,
        name=agent_utils.AgentName.router.name,
        response_format=router,
    )
    response = router_agent.invoke(state)

    # Format message for conversation history
    message_str = f"Route to {response['structured_response'].expert_agent} " \
                  f"agent with input " \
                  f"{response['structured_response'].agent_input}"

    # Adding question
    history_messages = f"**Human** query: {state['question']}\n\n"
    history_messages += \
        (f"**{agent_utils.AgentName.router.name.capitalize()} agent**: "
         f"{message_str}\n\n")

    return Command(
        goto=response["structured_response"].expert_agent,
        update={
            "messages": [
                AIMessage(
                    content=message_str,
                    name=agent_utils.AgentName.router.name,
                )
            ],
            "history_messages": history_messages,
            "next_agent": response["structured_response"].expert_agent,
            "next_input": response["structured_response"].agent_input
        },
    )


def reasoner_node(state: GraphState) -> Literal["generator"]:
    """
    Analyzes information and develops logical conclusions.

    The reasoner agent examines the query and any available information to form
    reasoned conclusions. It can work directly with the query or with research
    provided by the researcher agent. After reasoning, it passes control to the
    generator agent.

    Args:
        state (GraphState): The current state of the agent graph.

    Returns:
        Command: A directive to proceed to the generator node with
            reasoning results.
    """
    reasoner_agent = create_react_agent(
        model=agent_utils.get_model(model_name=_model_name),
        tools=[],
        prompt=prompt.REASONING.format(
            question=state["question"],
            history_messages=state["history_messages"]),
        name=agent_utils.AgentName.reasoner.name
    )

    # Prepare the human message, potentially including image
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

    response = reasoner_agent.invoke(state)
    last_message = agent_utils.get_last_message(response)

    # Update conversation history with reasoner's contribution
    history_messages = agent_utils.get_updated_history_messages(
        last_message, state["history_messages"],
        agent_utils.AgentName.reasoner.name)

    return Command(
        goto=agent_utils.AgentName.generator.name,
        update={
            "messages": [
                AIMessage(
                    content=last_message,
                    name=agent_utils.AgentName.reasoner.name,
                )
            ],
            "history_messages": history_messages,
            "next_agent": agent_utils.AgentName.generator.name,
            "next_input": last_message
        }
    )


def researcher_node(state: GraphState) -> Literal["reasoner"]:
    """
    Gathers information from external sources to answer the query.

    The researcher uses various tools (search, academic papers, wiki) to
    collect relevant information about the query. After gathering information,
    it passes control to the reasoner to analyze the findings.

    Args:
        state (GraphState): The current state of the agent graph.

    Returns:
        Command: A directive to proceed to the reasoner node with
            research results.
    """
    research_agent = create_react_agent(
        model=agent_utils.get_model(model_name=_model_name),
        tools=[
            tool.WikipediaTool(),
            tool.ArxivTool(),
            tool.TavilySearchTool()
        ],
        prompt=prompt.RESEARCH.format(
            history_messages=state["history_messages"]),
        name=agent_utils.AgentName.researcher.name
    )

    # Add the original question to the conversation
    state["messages"].append(HumanMessage(state["question"]))
    response = research_agent.invoke(state)

    # Update conversation history with research findings
    last_message = agent_utils.get_last_message(response)
    history_messages = agent_utils.get_updated_history_messages(
        last_message, state["history_messages"],
        agent_utils.AgentName.reasoner.name)

    return Command(
        goto=agent_utils.AgentName.reasoner.name,
        update={
            "messages": [
                AIMessage(
                    content=last_message,
                    name=agent_utils.AgentName.researcher.name,
                )
            ],
            "history_messages": history_messages,
            "next_agent": agent_utils.AgentName.reasoner.name,
            "next_input": last_message
        }
    )


def generator_node(state: GraphState) -> Literal["verifier", "END"]:
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
        next_agent = agent_utils.AgentName.verifier.name
        generator_prompt = prompt.GENERATOR_WITHOUT_FEEDBACK.format(
            history_messages=state["history_messages"])

    agent_name = "generator"
    generator_agent = create_react_agent(
        model=agent_utils.get_model(model_name=_model_name),
        tools=[],
        prompt=generator_prompt,
        name=agent_name
    )

    # Add the original question to the conversation
    state["messages"].append(HumanMessage(state["question"]))
    response = generator_agent.invoke(state)

    # Update conversation history
    last_message = agent_utils.get_last_message(response)
    history_messages = agent_utils.get_updated_history_messages(
        last_message, state["history_messages"],
        agent_utils.AgentName.generator.name)

    return Command(
        goto=next_agent,
        update={
            "messages": [
                AIMessage(
                    content=last_message,
                    name=agent_utils.AgentName.generator.name,
                )
            ],
            "history_messages": history_messages,
            "next_agent": agent_name,
            "next_input": last_message,
            "answer_feedback": "",
            # When there is no answer feedback, store as draft response
            "draft_answer": last_message if not state[
                "answer_feedback"] else "",
            # When there is answer feedback, store as final answer
            "final_answer": last_message if state["answer_feedback"] else "",
        }
    )


def verifier_node(state: GraphState) -> Literal["generator"]:
    """
    Evaluates the draft answer for accuracy, completeness, and quality.

    The verifier analyzes the draft answer produced by the generator
    and provides feedback for improvements. This feedback is then used by the
    generator to create a refined final answer.

    Args:
        state (GraphState): The current state of the agent graph.

    Returns:
        Command: A directive to return to the generator with verification
            feedback.
    """
    verifier_agent = create_react_agent(
        model=agent_utils.get_model(model_name=_model_name),
        tools=[],
        prompt=prompt.VERIFIER.format(
            answer=state["next_input"],
            history_messages=state["history_messages"]),
        name=agent_utils.AgentName.verifier.name
    )
    # Add the original question to the conversation
    state["messages"].append(HumanMessage(state["question"]))
    response = verifier_agent.invoke(state)

    # Update conversation history with verification feedback
    last_message = agent_utils.get_last_message(response)
    history_messages = agent_utils.get_updated_history_messages(
        last_message, state["history_messages"],
        agent_utils.AgentName.verifier.name)

    return Command(
        goto=agent_utils.AgentName.generator.name,
        update={
            "messages": [
                AIMessage(
                    content=last_message,
                    name=agent_utils.AgentName.verifier.name,
                )
            ],
            "history_messages": history_messages,
            "next_agent": agent_utils.AgentName.generator.name,
            "next_input": last_message,
            # Adding the feedback for the previous generator answer
            "answer_feedback": last_message,
        }
    )
