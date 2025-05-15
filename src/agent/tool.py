import os

import wikipedia
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import BaseTool
from langchain_experimental.utilities import PythonREPL
from src.tools.startup import logger


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
