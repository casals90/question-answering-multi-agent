import os
from typing import Any

from langchain_core.messages import AIMessage

from src.agent import node, workflow
from src.data import extract
from src.tools import audio, utils as tools_utils
from src.tools.startup import logger


class QuestionAnsweringAgent:
    def __init__(self, graph_config: dict[str, Any]) -> None:
        self._graph_config = graph_config
        self._graph = workflow.build_graph()

    def __str__(self):
        return "QuestionAnsweringAgent"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _pre_process_gaia_question(
            gaia_question: dict[str, str]) -> list[dict[str, Any]]:
        query = gaia_question["question"]
        if file_path := gaia_question.get("file_path"):
            if file_path.endswith(".py"):
                code = extract.read_file(file_path)
                # Adding the file content to the query
                query = f"{query}. Code: {code}"
                messages = [
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": query}
                     ]}
                ]
            elif file_path.endswith(".png"):
                logger.info(f"Adding png image")
                base64_image = tools_utils.image_to_base64(file_path)
                # messages = [
                #    {"role": "user", "content": [
                #        {"type": "text", "text": query},
                #        {"type": "image_url", "image_url": base64_image}
                #    ]}
                # ]
                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url",
                         "image_url": {"url": base64_image}}
                    ]}
                ]
            elif file_path.endswith(".mp3"):
                # Check if transcription.json file exists. If exists load it.
                # Otherwise, generate the transcription and store to disk
                transcription_file_path = \
                    f"{file_path.split('.')[0]}_transcription.json"
                if not os.path.exists(transcription_file_path):
                    transcription = audio.transcribe_audio_file(file_path)
                else:
                    transcription = extract.read_json_file(
                        transcription_file_path)

                query = (f"{query}. Audio transcription: "
                         f"{transcription['text'].strip()}")
                messages = [{"role": "user",
                             "content": [{"type": "text", "text": query}]}]
            else:
                raise ValueError(
                    f"File {gaia_question['filename']} "
                    f"extension is not supported.")
        else:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": query}]}]

        return messages

    def _answer_question(self, question: str, messages: list, **kwargs) -> str:
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
        messages = [
            {"role": "user",
             "content": [{"type": "text", "text": question}]}]

        return self._answer_question(question, messages, **kwargs)

    def answer_gaia_question(
            self, gaia_question: dict[str, str], **kwargs) -> str:
        messages = self._pre_process_gaia_question(gaia_question)
        return self._answer_question(
            gaia_question["question"], messages, **kwargs)
