# source code: https://github.com/octotools/octotools/tree/main/octotools/tools/url_text_extractor
"""
Compared to the source code, we add a custom HTTP header (User-Agent) to HTTP requests so the extractor can bypass some basic website anti-bot or block mechanisms that would otherwise prevent text extraction.
"""

import os, re, requests, sys
from datetime import datetime
from bs4 import BeautifulSoup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', '..')))
from opentools.core.base import BaseTool
class URL_Text_Extractor_Tool(BaseTool):
    DEFAULT_TEST_ARGS = {
        "tool_test": "url_text_extractor",
    }
    """URL_Text_Extractor_Tool
    ---------------------
    Purpose:
        A web text extraction tool that retrieves and extracts all text content from web pages using HTTP requests and HTML parsing. Processes web content to extract readable text while filtering out HTML markup, scripts, and other non-text elements.Limited to text extraction from web pages, does not support extraction from PDF files or other file types.

    Core Capabilities:
        - Extracts text content from web URLs using HTTP requests
        - Parses HTML content with BeautifulSoup
        - Filters out HTML markup and scripts to provide clean text output
        - Handles various web page formats and structures

    Intended Use:
        Use this tool when you need to extract text content from web pages, including filtering out HTML markup, scripts, and other non-text elements.

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
            name="URL_Text_Extractor_Tool",
            description="""A web text extraction tool that retrieves and extracts all text content from web pages using HTTP requests and HTML parsing. Processes web content to extract readable text while filtering out HTML markup, scripts, and other non-text elements.Limited to text extraction from web pages, does not support extraction from PDF files or other file types.CAPABILITIES: Extracts text content from web URLs using HTTP requests, parses HTML content with BeautifulSoup, filters out HTML markup and scripts to provide clean text output, handles various web page formats and structures. SYNONYMS: web text extractor, URL text scraper, web content extractor, HTML text parser, web page text extractor, website text extractor, web scraping tool, content extraction tool, web text parser, URL content extractor. EXAMPLES: 'Extract all text from this website URL', 'Get text content from this Wikipedia page', 'Extract readable text from this news article URL'.""",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL from which to extract text content"
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            strict=True,
            category="web_extraction",
            tags=["web_text_extraction", "url_processing", "html_parsing", "content_extraction", "web_scraping", "text_extraction", "web_content", "beautifulsoup", "http_requests", "web_tools"],
            limitation="Requires internet connection, may be blocked by anti-bot measures, some websites may not allow text extraction, JavaScript-heavy sites may not render properly, rate limiting may apply, depends on website structure and accessibility, limited to text extraction from web pages, does not support extraction from PDF files or other file types",
            agent_type="Browser_Extraction-Agent",
            accuracy= self.find_accuracy(os.path.join(os.path.dirname(__file__), 'test_result.json')),
            demo_commands= {
                "command": "reponse = tool.run(url='https://www.google.com')",
                "description": "Extract the text from the Google homepage"
            }
        )
        # Initialize a session for cookie persistence and better header handling
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            # Avoid brotli ("br") to reduce the chance of receiving content that the
            # runtime can't transparently decode.
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "DNT": "1",
        })
    
    
    def run(self, url):
        try:
            return self.extract_text_from_url(url)
        except Exception as e:
            return {"error": str(e), "success": False}

    def extract_text_from_url(self, url):
        """
        Extracts all text from the given URL.

        Parameters:
            url (str): The URL from which to extract text.

        Returns:
            str: The extracted text.
        """
        try:
            # Use session with pre-configured headers for better cookie handling and bot detection evasion
            # First, try to visit the domain root to establish session/cookies if needed
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                domain_root = f"{parsed.scheme}://{parsed.netloc}"
                # Make a lightweight HEAD request to establish session (don't follow redirects)
                self.session.head(domain_root, timeout=5, allow_redirects=False)
            except:
                pass  # Ignore errors in pre-request, proceed with main request
            
            # Add Referer header if we can determine it
            headers = {}
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"
            except:
                pass
            
            response = self.session.get(url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            # Check if the response is a PDF file
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                return {
                    "content": "Error: URL_Text_Extractor_Tool does not support PDF files. Please use Pdf_Extraction_Tool or Download_File_Tool followed by Pdf_Extraction_Tool for PDF content.",
                    "success": False
                }
            
            # Check if response is binary/not text
            if not response.encoding:
                # Try to detect encoding from headers or content
                if 'charset' in content_type:
                    encoding = content_type.split('charset=')[-1].split(';')[0].strip()
                else:
                    # Try to detect encoding from HTML meta tags or default to utf-8
                    encoding = 'utf-8'
            else:
                encoding = response.encoding
            
            # Decode content with proper encoding handling
            try:
                # Try to decode with detected/specified encoding
                html_content = response.content.decode(encoding, errors='replace')
            except (UnicodeDecodeError, LookupError):
                # Fallback to utf-8 with error replacement
                html_content = response.content.decode('utf-8', errors='replace')
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            
            # Check if the page requires JavaScript (common indicators)
            javascript_indicators = [
                'javascript is required',
                'please enable javascript',
                'enable javascript',
                'javascript must be enabled',
                'this site requires javascript',
                'noscript',
            ]
            
            text_lower = text.lower()
            has_javascript_warning = any(indicator in text_lower for indicator in javascript_indicators)
            
            # Also check if content is suspiciously minimal (likely JS-rendered page)
            # If text is very short (< 200 chars) and contains JS warnings, it's likely a SPA
            is_minimal_content = len(text.strip()) < 200
            
            # Check for noscript tags with content
            noscript_tags = soup.find_all('noscript')
            has_noscript_content = len(noscript_tags) > 0
            
            if has_javascript_warning and (is_minimal_content or has_noscript_content):
                return {
                    "content": f"Error: This webpage requires JavaScript to render content. URL_Text_Extractor_Tool only extracts static HTML and cannot execute JavaScript. Please use Browser_Interaction_Tool instead, which uses a real browser and can handle JavaScript-heavy websites like this one.",
                    "success": False
                }
            
            # text = text[:10000] # Limit the text to 10000 characters
            return {
                "url": url,
                "result": text,
                "success": True
            }
        except requests.RequestException as e:
            return {"content": f"Error fetching URL: {str(e)}", "success": False}
        except Exception as e:
            return {"content": f"Error extracting text: {str(e)}", "success": False}
        
    def test(self, tool_test: str = "url_text_extractor"):
        """Test the URL Text Extractor tool with various test samples, run 3 times, and save results in a JSON file."""
        try:
            # Load testbench data
            import json
            file_test = os.path.join(os.path.dirname(__file__), '..', 'test_file', 'data.json')
            with open(file_test, encoding='utf-8') as f:
                data = json.load(f)[tool_test]

            # Create test_results directory with timestamped filename
            tool_dir = os.path.dirname(__file__)
            test_results_dir = os.path.join(tool_dir, 'test_results')
            os.makedirs(test_results_dir, exist_ok=True)
            
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_result_{timestamp}.json"
            file_result = os.path.join(test_results_dir, filename)
            
            test_result = {}
            # Add metadata
            test_result['metadata'] = {
                "tool_name": self.name,
                "test_timestamp": datetime.now().isoformat(),
                "test_file": tool_test,
                "file_location": "url_text_extractor",
                "result_file": filename,
            }
            test_result['Test-File length'] = len(data)
            run_accuracy = {'run_1': 0, 'run_2': 0, 'run_3': 0}
            
            # Iterate over test cases
            for i, test in enumerate(data):
                question_result = {"id": test.get("id", f"url_extract_{i + 1}")}
                if 'url' in test:
                    question_result['query'] = test['url']
                if 'extracted_text' in test:
                    question_result['expected_answer'] = test['extracted_text']

                # Prepare parameters (exclude extracted_text, id, category)
                parameters = {k: v for k, v in test.items() if k not in ['extracted_text', 'id', 'category']}

                # Run and record result for each of 3 runs
                for j in range(1, 4):
                    run_result = {}
                    result = self.run(**parameters)
                    run_result['result'] = result

                    # If failed or none
                    if not result or (isinstance(result, dict) and result.get("success") == False):
                        run_result['accuracy'] = 0
                        run_result['tool_call_pass'] = False
                        question_result[f'run_{j}'] = run_result
                        continue
                    else:
                        run_result['tool_call_pass'] = True

                    # Calculate accuracy for this run using text comparison
                    if result.get("result") and test.get('extracted_text'):
                        extracted_text = re.sub(r"\s+", "", result['result'])
                        expected_text = re.sub(r"\s+", "", test['extracted_text'])
                        
                        # Check if extracted text matches expected text
                        if extracted_text == expected_text:
                            accuracy_score = 1
                        else:
                            accuracy_score = 0
                        
                        run_result['accuracy'] = accuracy_score
                        run_accuracy[f'run_{j}'] += accuracy_score
                    else:
                        run_result['accuracy'] = 0
                        run_accuracy[f'run_{j}'] += 0

                    question_result[f'run_{j}'] = run_result

                print(f"Finish query: {i + 1}")
                test_result[f'Q{i + 1}'] = question_result

            # Calculate and record overall accuracy for each run
            test_result['Final_Accuracy'] = {
                'run_1': run_accuracy['run_1'] * 100 / len(data) if data else 0,
                'run_2': run_accuracy['run_2'] * 100 / len(data) if data else 0,
                'run_3': run_accuracy['run_3'] * 100 / len(data) if data else 0
            }
            
            # Update metadata with final results
            test_result['metadata']['final_accuracy'] = test_result['Final_Accuracy']
            test_result['metadata']['total_questions'] = len(data)
            
            print(f"Accuracy: {test_result['Final_Accuracy']}")
            print(f"📁 Test result saved to: {file_result}")

            with open(file_result, "w", encoding="utf-8") as output_file:
                json.dump(test_result, output_file, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"❌ URL Text Extractor tool test failed: {e}")
            return False
        return True 
        
    def get_metadata(self):
        return super().get_metadata()
    
    def embed_tool(self):
        return super().embed_tool()
    
if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:

    cd octotools/tools/url_text_extractor
    python tool.py
    """

    # Get the directory of the current script                                                                                                                                                                                                                                                                                                           
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the URL_Text_Extractor_Tool
    tool = URL_Text_Extractor_Tool()
    tool.embed_tool()
    response = tool.run(url="https://www.benjerry.com/flavors/flavor-graveyard")
    print(response)
    # tool.test(tool_test="url_text_extractor")
