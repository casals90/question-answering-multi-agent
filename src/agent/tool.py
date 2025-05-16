import os
import tempfile

import wikipedia
from google import genai
from google.genai import types
from langchain.tools import BaseTool
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL

from src.tools import audio
from src.tools.startup import logger, settings


class WikipediaTool(BaseTool):
    """
    Tool that searches Wikipedia and returns article content.
    """
    name: str = "wikipedia_search"
    description: str = \
        "Search Wikipedia for the query and return the top articles."

    def _run(self, query: str) -> str:
        """
        Search Wikipedia for the query and return the articles.

        Args:
            query (str): Wikipedia search query

        Returns:
            str: The Wikipedia search results
        """
        # Search for the page
        try:
            search_results = wikipedia.search(query, results=10)
            if not search_results:
                return "No results found on Wikipedia for that query."

            results = "Top Wikipedia articles related to query:\n"
            # Get the summary of the first result
            for search_result in search_results:
                page_title = search_result
                try:
                    page = wikipedia.page(page_title)
                except Exception as e:
                    continue
                page_content = page.content
                results = f"{results}{page_title}\n{page_content}\n\n"

            return results
        except wikipedia.DisambiguationError as e:
            return (
                    f"The query was ambiguous. Possible options include:\n" +
                    ", ".join(e.options[:10])
            )

    def _arun(self, query: str) -> str:
        """Async implementation of the tool."""
        # For most tools, we can just call the synchronous version
        return self._run(query)


class ArxivTool(ArxivQueryRun):
    """Wrapper around ArxivQueryRun to maintain consistent interface."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "arxiv_search"
        self.description = \
            "Search arXiv for academic papers and return the results."


class TavilySearchTool(TavilySearchResults):
    """
    Wrapper around TavilySearchResults to maintain consistent interface.
    """

    def __init__(self) -> None:
        super().__init__(api_key=os.environ["TAVILY_API_KEY"])
        self.name = "tavily_search"
        self.description = \
            "Search the web using Tavily and return the results."


class PythonReplTool(BaseTool):
    """
    Tool that executes Python code in a REPL environment.
    """

    name: str = "python_repl"
    description: str = \
        "Execute the provided Python code and return the output."

    def _run(self, code: str) -> str:
        """
        Execute the provided Python code and return the output.

        Args:
            code (str): The python code to execute.

        Returns:
            str: The execution results or error message.
        """
        logger.info(f"Generated code {code}\n\n")
        try:
            # Create a new Python REPL instance
            repl = PythonREPL()
            result = repl.run(code)
            logger.info(f"Execution result {result}")
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"

        result_str = (f"Successfully executed:\n"
                      f"```python\n{code}\n```\nStdout: {result}")
        return result_str + ("\n\nIf you have completed all tasks, "
                             "respond with FINAL ANSWER.")

    def _arun(self, code: str) -> str:
        """
        Asynchronous version of _run. For this tool, we simply call
        the synchronous version.

        Args:
            code (str): The python code to execute.

        Returns:
            str: The execution results or error message.
        """
        return self._run(code)


class GetYoutubeUrlTranscription(BaseTool):
    """
    Tool that downloads audio from a YouTube video and transcribes it.
    """
    name: str = "youtube_transcription"
    description: str = \
        "Given a YouTube URL, download the audio and transcribe it to text."

    def _run(self, url: str) -> str:
        """
        Download audio from YouTube URL and transcribe it.

        Args:
            url (str): YouTube video URL

        Returns:
            str: The transcription of the video's audio
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download the audio from the YouTube URL
                file_path = audio.download_youtube_audio(
                    url, output_folder=temp_dir)

                # Verify the file exists
                if not os.path.exists(file_path):
                    return f"Error: Failed to download audio from {url}"

                # Transcribe the audio file
                transcription = audio.transcribe_audio_file(file_path)

                # Return the transcription text
                if isinstance(transcription, dict) and "text" in transcription:
                    return transcription["text"]
                else:
                    return str(transcription)

        except Exception as e:
            return (f"An error occurred while processing "
                    f"the YouTube URL: {str(e)}")

    async def _arun(self, url: str) -> str:
        """Async implementation of the tool."""
        # For most tools, we can just call the synchronous version
        return self._run(url)


class YoutubeVideoQuery(BaseTool):
    """
    Tool that uses Google's Gemini model to analyze a YouTube video and answer
    questions about it.
    """
    name: str = "youtube_query"
    description: str = \
        ("Given a YouTube URL and a question, analyze the video content using "
         "Gemini and return the answer.")

    def _run(self, input_dict: dict) -> str:
        """
        Analyze a YouTube video and answer a question about its visual content.

        Args:
            input_dict (dict): Dictionary containing 'url' (YouTube video URL)
            and 'query' (question about the video)

        Returns:
            str: The answer to the query about the video content
        """
        try:
            # Extract URL and query from the input dictionary
            if not isinstance(input_dict, dict):
                return ("Input must be a dictionary"
                        " with 'url' and 'query' keys.")

            url = input_dict.get('url')
            query = input_dict.get('query')

            if not url or not query:
                return "Both 'url' and 'query' must be provided."

            # Initialize the Gemini client
            client = genai.Client()

            # Create the request to Gemini
            # 'models/gemini-2.0-flash'
            response = client.models.generate_content(
                model=f"models/{settings['GOOGLE_MODEL_ID']}",
                contents=types.Content(
                    parts=[
                        types.Part(
                            file_data=types.FileData(file_uri=url)
                        ),
                        types.Part(text=query)
                    ]
                )
            )

            # Return the response text
            return response.text

        except Exception as e:
            return (f"An error occurred while analyzing "
                    f"the YouTube video: {str(e)}")

    async def _arun(self, input_dict: dict) -> str:
        """Async implementation of the tool."""
        # For most tools, we can just call the synchronous version
        return self._run(input_dict)
