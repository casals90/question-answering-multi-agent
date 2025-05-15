import os

import wikipedia
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing_extensions import Annotated


@tool
def wikipedia_tool(query: str) -> str:
    """
    Search Wikipedia for the query and return the article top 5 articles.

    Parameters:
        query (str): wikipedia's query

    Returns:
        str: The wikipedia
    """
    # Search for the page
    try:
        search_results = wikipedia.search(query, results=5)
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


@tool
def python_repl_tool(
        code: Annotated[
            str, "The python code to execute to generate your chart."],
):
    """Execute the provided Python code and return the output."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
            result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


@tool
def excel_processor(file_path: str, query: str = None) -> str:
    """Process an Excel file and return data based on query"""
    if not os.path.exists(file_path):
        return "File not found"

    try:
        # Read Excel file
        df = pd.read_excel(file_path)

        # If no specific query, return basic info
        if not query:
            return f"Excel file contains {len(df)} rows and {len(df.columns)} columns. Columns: {', '.join(df.columns.tolist())}"

        # Execute query using pandas eval if applicable
        # For safety, we'd want more complex parsing here
        return str(eval(f"df.{query}"))
    except Exception as e:
        return f"Error processing Excel file: {str(e)}"


arxiv_tool = ArxivQueryRun()
tavily_search_tool = TavilySearchResults(
    max_results=5,
    include_raw_content=False,
    include_images=False,
)
