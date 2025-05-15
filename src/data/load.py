import json
import os
import pathlib
import requests
from typing import Any


def save_json_file(
        data: [dict[str, Any], list[Any]],
        file_path: str,
        indent: int = 2,
        ensure_ascii: bool = False
) -> str:
    """
    Save data as a JSON file.

    Args:
        data (Dict[str, Any] or List[Any]): The data to save as JSON
        file_path (str): Path where the JSON file will be saved
        indent (int): Number of spaces for indentation in the JSON file
        ensure_ascii (bool): If False, non-ASCII characters are
                             output as-is (default). If True,
                             non-ASCII chars are escaped

    Returns:
        str: Path where the JSON file was saved

    Raises:
        ValueError: If the data is not JSON serializable
        IOError: If there's an error writing to the file
    """
    try:
        path = pathlib.Path(file_path)

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write data to JSON file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

        return str(path)

    except TypeError as e:
        raise ValueError(f"Data is not JSON serializable: {e}")
    except IOError as e:
        raise IOError(f"Error writing JSON to file: {e}")


def submit_answers(answers: list[dict[str, str]]) -> str:
    submit_data = {
        "username": os.environ["HF_USERNAME"],
        "agent_code": os.environ["HF_AGENT_CODE"],
        "answers": answers
    }
    hf_url_submit = "https://agents-course-unit4-scoring.hf.space/submit"
    response = requests.post(hf_url_submit, data=json.dumps(submit_data))

    return response.text
