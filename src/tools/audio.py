import os
from pathlib import Path
from typing import Any

import whisper
import yt_dlp

from src.data import load
from src.tools.startup import logger
from src.tools import utils as tools_utils


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


def download_youtube_audio(
        url: str,
        output_folder: str,
        audio_format: str = 'mp3',
        audio_quality: str = '192',
        verbose: bool = True
) -> str:
    """
    Download audio from a YouTube video.

    Args:
        url (str): URL of the YouTube video to download audio from
        audio_format (str): Desired audio format (e.g., 'mp3', 'm4a', 'wav'),
            defaults to 'mp3'
        audio_quality (str): Audio quality in kbps (e.g., '128', '192', '320'),
            defaults to '192'
        output_folder (str): Folder path where the audio file will be saved,
            defaults to None (current directory)
        verbose (bool): Whether to show download progress, defaults to True

    Returns:
        str: The file path of the downloaded audio file
    """

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    unique_id = tools_utils.generate_unique_id()
    file_path = os.path.join(output_folder, f"{unique_id}")

    # Options for yt_dlp
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
            'preferredquality': audio_quality,
        }],
        'outtmpl': file_path,
        'quiet': not verbose,
        # Extract info before downloading
        'noplaylist': True,
    }

    # Get video info first to determine output file path
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            # Download the video
            ydl.download([url])

        # Adding mp3 extension
        file_path = f"{file_path}.mp3"
        logger.info(f"Download completed. Audio saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        file_path = None

    return file_path
