# source code: https://github.com/octotools/octotools/blob/main/octotools/tools/url_text_extractor/tool.py

import os, sys, requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
from bs4 import BeautifulSoup

class Url_Text_Extractor_Octotools_Tool(BaseTool):
    """Url_Text_Extractor_Octotools_Tool
    ---------------------
    Purpose:
        A tool that extracts all text content from a given URL webpage.

    Core Capabilities:
        - Extracts text content from web URLs
        - Parses HTML content
        - Handles arXiv PDF URLs by converting to abstract pages
        - Limits text extraction to prevent excessive output
        - Formats extracted text with clean formatting

    Intended Use:
        Use this tool when you need to extract text content from a given URL webpage.

    Limitations:
        - Requires internet connection
        - May be blocked by anti-bot measures
        - Some websites may not allow text extraction
        - JavaScript-heavy sites may not render properly
        - Rate limiting may apply
        - Depends on website structure and accessibility
        - Limited to text extraction from web pages
        - Does not support extraction from PDF files or other file types
    """

    def __init__(self):
        super().__init__(
            type='function',
            name="Url_Text_Extractor_Octotools_Tool",
            description="""A tool that extracts all text content from a given URL webpage. CAPABILITIES: Extract text content from web URLs, parse HTML content, handle arXiv PDF URLs by converting to abstract pages, limit text extraction to prevent excessive output, format extracted text with clean formatting. SYNONYMS: URL text extractor, webpage text extractor, web content extractor, HTML text parser, URL content scraper, web text scraper, URL text parser. EXAMPLES: 'Extract all text from https://example.com', 'Get text content from Wikipedia page about Python programming language', 'Extract text from arXiv paper abstract page'.""",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL from which to extract text."
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            strict=False,
            category="web_extraction",
            tags=["url_extraction", "web_scraping", "text_extraction", "html_parsing", "web_content", "content_extraction", "url_scraper", "web_text"],
            limitation="Requires internet connection, depends on website availability and structure, text extraction limited to 10000 characters, may fail with protected or dynamic content, HTML parsing may break if website structure changes, arXiv PDF URLs are automatically converted to abstract pages.",
            agent_type="File_Extraction-Agent",
            demo_commands={
                "command": 'execution = tool.run(url="https://example.com")',
                "description": "Extract all text from the example.com website."
            }
        )

    def extract_text_from_url(self, url):
        """
        Extracts all text from the given URL.

        Parameters:
            url (str): The URL from which to extract text.

        Returns:
            str: The extracted text.
        """
        url = url.replace("arxiv.org/pdf", "arxiv.org/abs")

        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            text = text[:10000] # Limit the text to 10000 characters
            return text
        except requests.RequestException as e:
            return f"Error fetching URL: {str(e)}"
        except Exception as e:
            return f"Error extracting text: {str(e)}"

    def run(self, url):
        """
        Extracts text from the provided URL and returns the results.

        Parameters:
            url (str): The URL from which to extract text.

        Returns:
            dict: A dictionary containing the URL and extracted text.
        """
        extracted_text = self.extract_text_from_url(url)
        return {
            "url": url,
            "result": extracted_text,
            "success": True
        }

    def execute(self, url):
        """Legacy method name for backward compatibility. Use run() instead."""
        return self.run(url)
    
    def get_metadata(self):
        """
        Returns the metadata for the Url_Text_Extractor_Octotools_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        return super().get_metadata()

if __name__ == "__main__":
    tool = Url_Text_Extractor_Octotools_Tool()
    print(tool.run(url="https://www.cnn.com/"))