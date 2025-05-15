import json
import os
import pathlib
from typing import Any

import requests

from src.tools.startup import logger


def get_questions(output_path: str) -> list[dict[str, Any]]:
    """
    Downloads all questions from the API and optionally downloads associated
        files.

    Parameters:
    - output_path (str, optional): If provided, files associated with questions
        will be downloaded to this path

    Returns:
    - list: A list of question dictionaries, or empty list if the request
        failed
    """
    # Define API URL
    base_url = "https://agents-course-unit4-scoring.hf.space"
    url_questions = f"{base_url}/questions"

    try:
        # Send a GET request to fetch the questions
        response = requests.get(url_questions)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            questions = response.json()
            logger.info(f"Successfully retrieved {len(questions)} questions")

            # If output_path is provided, download files for each question
            # if they exist
            if output_path:
                for question in questions:
                    if question.get('file_name'):
                        logger.info(f"Found file for task "
                                    f"{question.get('task_id')}: "
                                    f"{question.get('file_name')}")
                        file_path = get_question_file(question, output_path)
                        question["file_path"] = file_path
        else:
            logger.info(f"Failed to retrieve questions. "
                        f"Status code: {response.status_code}")
            questions = []

    except Exception as e:
        logger.error(f"An error occurred while retrieving questions: {str(e)}")
        questions = []

    return questions


def get_question_file(
        question: dict[str, Any],
        output_path: str,
        override: bool = False
) -> str:
    """
    Downloads a file based on the question dictionary and saves it to
    the output_path.

    Parameters:
    - question (dict): A dictionary containing task_id, question, level,
        and file_name
    - output_path (str): The path where the file should be saved
    - override (bool): Override if file exists.

    Returns:
    - str: The output file path if exists. Otherwise, empty str.
    """
    # Extract necessary information from the question dictionary
    task_id = question.get('task_id')
    file_name = question.get('file_name')

    # Check if there's a file to download
    if not file_name:
        logger.info(f"No file_name found for task {task_id}")
        return ""

    # Define API URLs
    base_url = "https://agents-course-unit4-scoring.hf.space"
    url_files = f"{base_url}/files/{task_id}"

    # Define the full path for saving the file
    task_dir = os.path.join(output_path, str(task_id))
    file_path = os.path.join(task_dir, file_name)
    if not os.path.exists(file_path) or override:
        try:
            # Send a GET request to download the file
            response = requests.get(url_files)

            # Check if the request was successful
            if response.status_code == 200:
                # Create the task directory if it doesn't exist
                pathlib.Path(task_dir).mkdir(parents=True, exist_ok=True)

                # Save the file
                with open(file_path, 'wb') as f:
                    f.write(response.content)

                logger.info(f"File '{file_name}' successfully downloaded "
                            f"to {file_path}")
            else:
                logger.info(f"Failed to download file. "
                            f"Status code: {response.status_code}")

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
    else:
        logger.info("Skipping download. The file already exists.")

    return file_path


def read_file(file_path: str) -> str:
    """
    Read a  file and return its content.

    Parameters:
        file_path (str):

    Returns:
        str:
    """
    with open(file_path, "r") as f:
        return f.read()


def read_json_file(file_path: str) -> [dict[str, Any], list[Any]]:
    """
    Read a JSON file and return the data.

    Args:
        file_path (str): Path to the JSON file to read

    Returns:
        dict[str, Any] or list[Any]: The loaded JSON data

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        IOError: If there's an error reading the file
    """
    try:
        # Convert file_path to Path object
        path = pathlib.Path(file_path)

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        # Read and parse the JSON file
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON format: {e.msg}", e.doc,
                                   e.pos)
    except IOError as e:
        raise IOError(f"Error reading JSON file: {e}")
