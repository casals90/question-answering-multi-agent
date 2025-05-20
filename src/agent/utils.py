import enum
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.tools.startup import logger, settings


class ModelName(enum.Enum):
    """
    Enumeration of supported model providers.

    This enum defines the valid model providers that can be used with the
    get_model factory function. Each enum value corresponds to a specific
    model provider's identifier string.

    Attributes:
        google: Identifier for Google Generative AI models (Gemini)
        openai: Identifier for OpenAI models (GPT)
    """
    google = "google"
    openai = "openai"


def get_last_message(model_response: dict[str, Any]) -> str:
    """
    Extract the content of the last message from a model response.

    This function safely extracts the text content from the last message in a
    model response dictionary. It handles potential exceptions if the message
    structure doesn't match expectations.

    Args:
        model_response (dict[str, Any]): Dictionary containing the model's
            complete response, expected to have a 'messages' key with a list
            of message objects.

    Returns:
        str: The text content of the last message.

    Raises:
        ValueError: If no messages are found in the response.
    """
    try:
        last_message = model_response["messages"][-1].content
    except IndexError as ie:
        logger.warning(f"There is not last message {ie}")
        raise ValueError("There is not messages")

    return last_message


def get_updated_history_messages(
        last_message: str, history_messages: str, agent_name: str) -> str:
    """
    Append a new agent message to the conversation history string.

    This function formats an agent's message with proper attribution and
    adds it to the ongoing conversation history. The formatting includes
    the agent name (capitalized) and Markdown formatting for readability.

    Args:
        last_message (str): The content of the message to append.
        history_messages (str): The existing conversation history string.
        agent_name (str): The name of the agent who produced the message.

    Returns:
        str: Updated conversation history with the new message appended.
    """
    history_messages += \
        f"**{agent_name.capitalize()} agent**: {last_message}\n\n"

    return history_messages


def get_google_model(
        model_id: str,
        temperature: int = 0,
        **kwargs
) -> ChatGoogleGenerativeAI:
    """
    Initialize and configure a Google Generative AI chat model.

    This function creates a configured instance of the Google Generative AI
    chat model with specified parameters. It serves as a factory function
    to standardize model initialization across the application.

    Args:
        model_id (str): The identifier for the specific Google AI model to use
            (e.g., "gemini-pro", "gemini-ultra").
        temperature (int, optional): Controls randomness in output generation.
            Lower values (like 0) make output more deterministic and focused,
            while higher values increase creativity and variability.
            Defaults to 0 for consistent responses.
        **kwargs: Additional keyword arguments passed directly to the
            ChatGoogleGenerativeAI constructor for further customization
            (e.g., top_p, top_k, max_output_tokens).

    Returns:
        ChatGoogleGenerativeAI: Configured instance of the Google AI chat model
            ready to use for generating responses.

    Note:
        The default temperature of 0 prioritizes deterministic, predictable
        responses, which is typically preferred for agent-based systems where
        consistency is important.
    """
    return ChatGoogleGenerativeAI(
        model=model_id, temperature=temperature, **kwargs)


def get_openai_model(
        model_id: str,
        temperature: int = 0,
        **kwargs
) -> ChatOpenAI:
    """
    Initialize and configure an OpenAI chat model.

    This function creates a configured instance of the OpenAI chat model
    with specified parameters.

    Args:
       model_id (str): The identifier for the specific OpenAI model to use
           (e.g., "gpt-4", "gpt-3.5-turbo").
       temperature (int, optional): Controls randomness in output generation.
           Lower values (like 0) make output more deterministic and focused,
           while higher values increase creativity and variability.
           Defaults to 0 for consistent responses.
       **kwargs: Additional keyword arguments passed directly to the
           ChatOpenAI constructor for further customization
           (e.g., top_p, max_tokens, presence_penalty).

    Returns:
       ChatOpenAI: Configured instance of the OpenAI chat model
           ready to use for generating responses.
       """
    return ChatOpenAI(model=model_id, temperature=temperature, **kwargs)


def get_model(
        model_name: str = ModelName.google.value,
        temperature: int = 0,
        **kwargs
) -> ChatGoogleGenerativeAI | ChatOpenAI:
    """
    Factory function to create and configure a chat model from supported
    providers.

    This function serves as a unified interface for creating chat models from
    different providers. It delegates to the appropriate provider-specific
    function based on the model_name parameter.

    Args:
        model_name (str, optional): The provider of the model. Must be one of
            the values defined in ModelName enum. Defaults to "google".
        temperature (int, optional): Controls randomness in output generation.
            Lower values (like 0) make output more deterministic.
            Defaults to 0 for consistent responses.
        **kwargs: Additional keyword arguments passed to the specific model
            constructor for further customization.

    Returns:
        Union[ChatGoogleGenerativeAI, ChatOpenAI]: Configured instance of the
            specified chat model ready to use for generating responses.

    Raises:
        ValueError: If the provided model_name is not a valid provider
            identifier.
    """

    if model_name == ModelName.google.value:
        chat_model = get_google_model(
            settings["GOOGLE_MODEL_ID"], temperature, **kwargs)
    elif model_name == ModelName.openai.value:
        chat_model = get_openai_model(
            settings["OPENAI_MODEL_ID"], temperature, **kwargs)
    else:
        raise ValueError(f"The {model_name} is not valid.")

    return chat_model


def create_router(agents: list[str]) -> type[BaseModel]:
    """
    Creates a Pydantic model for validating routing decisions.

    This function dynamically generates a Pydantic model class that
    enforces validation rules for routing between agents. The expert_agent
    field can only contain values from the provided options list.

    Args:
        agents (list[str]): List of valid agent names for routing.

    Returns:
        type[BaseModel]: A Pydantic model class with next_agent
            field validation.
    """

    class Router(BaseModel):
        expert_agent: str = Field(
            description="The expert agent to route", enum=agents)
        agent_input: str = Field(
            description="The input of expert agent"
        )

    return Router
