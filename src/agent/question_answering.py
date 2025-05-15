import os
from typing import Any

from langchain_core.messages import AIMessage

from src.agent import node, workflow
from src.data import extract
from src.tools import audio, utils as tools_utils
from src.tools.startup import logger


class QuestionAnsweringAgent:
    """
    A class that handles question answering using a graph-based workflow.

    This agent processes various input types (text, images, audio) and uses a
    graph-based workflow to generate answers to questions.

    Attributes:
        _graph_config (dict[str, Any]): Configuration parameters for the
            graph workflow.
        _graph: The computational graph used for processing questions and
            generating answers.
    """

    def __init__(self, graph_config: dict[str, Any]) -> None:
        """
        Initialize the QuestionAnsweringAgent with the given graph
        configuration.

        Args:
            graph_config (dict[str, Any]): Configuration dictionary for the
                graph workflow.
        """
        self._graph_config = graph_config
        self._graph = workflow.build_graph()

    def __str__(self) -> str:
        """
        Return a string representation of the QuestionAnsweringAgent.

        Returns:
            str: The name of the agent.
        """
        return "QuestionAnsweringAgent"

    def __repr__(self) -> str:
        """
        Return the official string representation of the
        QuestionAnsweringAgent.

        Returns:
            str: The string representation from __str__.
        """
        return self.__str__()

    @staticmethod
    def _pre_process_gaia_question(
            gaia_question: dict[str, str]) -> list[dict[str, Any]]:
        """
        Preprocess a GAIA question by handling different file types and
        formats.

        This method processes the question based on the file type provided:
        - Python files: Reads code content and adds it to the query
        - PNG images: Converts image to base64 and includes it in the message
        - MP3 audio: Transcribes audio or reads existing transcription and
            includes it in the query

        Args:
            gaia_question (dict[str, str]): A dictionary containing the
                question and optional file path.
                Expected keys:
                - "question": The text of the question
                - "file_path" (optional): Path to an associated file

        Returns:
            list[dict[str, Any]]: Formatted messages ready for processing by
                the agent.

        Raises:
            ValueError: If an unsupported file extension is provided.
        """
        query = gaia_question["question"]
        if file_path := gaia_question.get("file_path"):
            if file_path.endswith(".py"):
                logger.info(f"Adding python code to messages.")
                code = extract.read_file(file_path)
                # Adding the file content to the query
                query = f"{query}.\n### Code:\n{code}"
                messages = [
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": query}
                     ]}
                ]
            elif file_path.endswith(".png"):
                logger.info(f"Adding png image to messages.")

                base64_image = tools_utils.image_to_base64(file_path)
                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url",
                         "image_url": {"url": base64_image}}
                    ]}
                ]
            elif file_path.endswith(".mp3"):
                logger.info(f"Adding mp3 transcription.")
                # Check if transcription.json file exists. If exists load it.
                # Otherwise, generate the transcription and store to disk
                transcription_file_path = \
                    f"{file_path.split('.')[0]}_transcription.json"
                if not os.path.exists(transcription_file_path):
                    transcription = audio.transcribe_audio_file(file_path)
                else:
                    transcription = extract.read_json_file(
                        transcription_file_path)

                query = (f"{query}.\n### Audio transcription: "
                         f"{transcription['text'].strip()}")
                messages = [{"role": "user",
                             "content": [{"type": "text", "text": query}]}]
            elif file_path.endswith(".xlsx"):
                logger.info(f"Adding excel filepath.")
                query = f"{query}\n###File path:\n{file_path}"
                messages = [
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": query}
                     ]}
                ]
            else:
                raise ValueError(
                    f"File {gaia_question['filename']} "
                    f"extension is not supported.")
        else:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": query}]}]

        return messages

    def _answer_question(self, question: str, messages: list, **kwargs) -> str:
        """
        Internal method to process a question through the graph workflow.

        This method initializes a GraphState with the question and messages,
        runs the graph workflow, and returns the final answer.

        Args:
            question (str): The original question text.
            messages (list): Formatted messages ready for processing.
            **kwargs: Additional arguments to pass to the graph.

        Returns:
            str: The answer to the question, or an empty string if
                processing failed.
        """
        # Initialize the GraphState
        state = node.GraphState(
            question=question,
            messages=messages,
            history_messages="",
            next_agent="",
            next_input="",
            answer_feedback="",
            draft_answer="",
            final_answer="",
            image="",
        )

        response = {}
        try:
            for response in self._graph.stream(
                    state, self._graph_config, **kwargs):
                last_message = response["messages"][-1]
                if isinstance(last_message, AIMessage):
                    logger.info(f"{last_message.name.capitalize()}:")
                    logger.info("-" * 20)
                    logger.info(f"{last_message.content}\n\n")

            # Get the final answer
            answer = response.get("final_answer", "")
        except Exception as e:
            logger.error(
                f"Exception {e} answering the question {state['question']}")
            answer = ""

        return answer

    def answer_question(self, question: str, **kwargs) -> str:
        """
        Process a simple text question and return an answer.

        Args:
            question (str): The question to answer.
            **kwargs: Additional arguments to pass to the graph.

        Returns:
            str: The answer to the question.
        """
        messages = [
            {"role": "user",
             "content": [{"type": "text", "text": question}]}]

        return self._answer_question(question, messages, **kwargs)

    def answer_gaia_question(
            self, gaia_question: dict[str, str], **kwargs) -> str:
        """
        Process a GAIA question with potential file attachments and return
        an answer.

        GAIA questions may include references to files like code, images, or
        audio which require special preprocessing before answering.

        Args:
            gaia_question (dict[str, str]): A dictionary containing the
            question and optional file path.
                Expected keys:
                - "question": The text of the question
                - "file_path" (optional): Path to an associated file
            **kwargs: Additional arguments to pass to the graph.

        Returns:
            str: The answer to the question.
        """
        messages = self._pre_process_gaia_question(gaia_question)
        return self._answer_question(
            gaia_question["question"], messages, **kwargs)
