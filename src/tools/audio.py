import os
import whisper

from pathlib import Path
from src.data import load
from src.tools.startup import logger
from typing import Any


def transcribe_audio_file(
        file_path: str,
        model_name: str = "turbo",
        language: str | None = None,
        **kwargs: Any
) -> dict[str, Any]:
    """
    Transcribe audio from a file path using OpenAI's Whisper model.
    In addition, it stores the transcription into a disk.

    Args:
        file_path (str): Path to the audio file to transcribe
        model_name (str): Whisper model to use (tiny, base,
            small, medium, large)
        language (str | None): Language code for transcription (e.g.,
            "en" for English). If None, Whisper will auto-detect the language
        **kwargs: Additional arguments to pass to the Whisper model

    Returns:
        dict[str, Any]: Transcription text or detailed
            JSON with segments and metadata

    Raises:
        ValueError: If the file path or model parameters are invalid
        FileNotFoundError: If the audio file doesn't exist
        RuntimeError: If there's an issue with the transcription process
    """
    # Validate input parameters
    if not file_path:
        raise ValueError("File path cannot be empty")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        logger.info(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)

        # Prepare transcription options
        transcribe_options = kwargs.copy()
        if language:
            transcribe_options["language"] = language

        # Perform transcription
        logger.info(f"Starting transcription of {file_path}")
        result = model.transcribe(str(file_path), **transcribe_options)

        # Create output filename based on input file
        output_file = os.path.join(
            file_path.parent, f"{file_path.stem}_transcription.json")

        # Ensure the output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving transcription to {output_file}")
        load.save_json_file(result, str(output_path))

        return result

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise RuntimeError(f"Transcription failed: {e}")
