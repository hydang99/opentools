# source code: https://github.com/octotools/octotools/blob/main/octotools/tools/nature_news_fetcher/tool.py
"""
Update: Added function to count the number of news articles published in a given year.
This enables efficient extraction of annual publication statistics from Nature's news archive, supporting research and time series analysis.
"""

import os, sys, requests, time, traceback
from bs4 import BeautifulSoup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
class Nature_News_Fetcher_Tool(BaseTool):
    """Nature_News_Fetcher_Tool
    ---------------------
    Purpose:
        A comprehensive tool for fetching and analyzing news articles from Nature's website. Supports year-based filtering, article counting, and detailed article information extraction including titles, URLs, descriptions, authors, publication dates, and image URLs. Ideal for research, content analysis, academic purposes, and statistical analysis of publication patterns.

    Core Capabilities:
        - Fetches news articles from Nature website with year-based filtering
        - Provides article counting and statistics
        - Extracts detailed article information (titles, URLs, descriptions, authors, dates, images)
        - Supports comprehensive data collection and statistical analysis of publication patterns

    Intended Use:
        Use this tool when you need to fetch and analyze news articles from Nature's website, including year-based filtering, article counting, and detailed article information extraction.

    Limitations:
        - May not handle complex news article information extraction

    """
    def __init__(self):
        super().__init__(
            type='function',
            name="Nature_News_Fetcher_Tool",
            description="""A comprehensive tool for fetching and analyzing news articles from Nature's website. Supports year-based filtering, article counting, and detailed article information extraction including titles, URLs, descriptions, authors, publication dates, and image URLs. Ideal for research, content analysis, academic purposes, and statistical analysis of publication patterns. CAPABILITIES: Fetches news articles from Nature website with year-based filtering, provides article counting and statistics, extracts detailed article information (titles, URLs, descriptions, authors, dates, images), supports comprehensive data collection and statistical analysis of publication patterns. SYNONYMS: Nature article fetcher, publication data collector, Nature news crawler, scientific publication counter, research data extractor. EXAMPLES: 'Fetch 50 latest articles from the first 3 pages of Nature news', 'Get the total count of articles published in 2022 for statistical analysis', 'Analyze article distribution across all years for comprehensive statistical analysis'.""",
            parameters={
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "Year filter (e.g., 2023) to retrieve articles from a specific publication year"
                    },
                    "num_articles": {
                        "type": ["integer", "null"],
                        "description": "Number of articles to fetch. Use null to fetch all available articles without limit"
                    },
                    "max_pages": {
                        "type": ["integer", "null"],
                        "description": "Maximum number of pages to crawl. Use null to fetch from all available pages"
                    },
                    "count_only": {
                        "type": "boolean",
                        "description": "When true, returns only article counts by year instead of full article details. Useful for statistical analysis"
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
            strict=False,
            category="data_collection",
            tags=["nature_news", "article_fetcher", "data_collection", "publication_analysis", "news_scraping", "statistical_analysis"],
            limitation="Requires internet connection, limited to Nature website data, may have rate limiting, historical data availability varies, depends on website structure changes",
            agent_type="Search-Agent",
            demo_commands= {
                "command": "reponse = tool.run(num_articles=50, max_pages=3, year=2023, count_only=False)",
                "description": "Fetch 50 latest articles from the first 3 pages of Nature news"
            }            
        )
        self.base_url = "https://www.nature.com/nature/articles"

    def fetch_page(self, page_number, year=None):
        """
        Fetches a single page of news articles from Nature's website.

        Parameters:
            page_number (int): The page number to fetch.
            year (int, optional): Filter articles by specific year.

        Returns:
            str: The HTML content of the page, or None if page not found.
        """
        params = {
            "searchType": "journalSearch",
            "sort": "PubDate",
            "type": "article",
            "page": str(page_number)
        }
        # Add year filter if specified
        if year:
            params["year"] = str(year)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        try:
            response = requests.get(self.base_url, params=params, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Page not found - this is normal when we've reached the end
                return None
            else:
                # Re-raise other HTTP errors
                raise
        except Exception as e:
            # Handle other exceptions
            raise

    def parse_articles(self, html_content, target_year=None):
        """
        Parses the HTML content and extracts article information.

        Parameters:
            html_content (str): The HTML content of the page.
            target_year (int, optional): Filter articles by specific year.

        Returns:
            list: A list of dictionaries containing article information.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        articles_section = soup.find('section', id='new-article-list')
        if not articles_section:
            return []

        articles = []
        # Try different selectors to find all articles
        article_elements = articles_section.find_all('article', class_='c-card')
        if not article_elements:
            # Try alternative selectors
            article_elements = articles_section.find_all('article')
        
        for article in article_elements:
            title_elem = article.find('h3', class_='c-card__title')
            title = title_elem.text.strip() if title_elem else "No title found"
            
            url_elem = title_elem.find('a') if title_elem else None
            url = "https://www.nature.com" + url_elem['href'] if url_elem and 'href' in url_elem.attrs else "No URL found"
            
            description_elem = article.find('div', {'data-test': 'article-description'})
            description = description_elem.text.strip() if description_elem else "No description available"
            
            authors_elem = article.find('ul', {'data-test': 'author-list'})
            authors = [author.text.strip() for author in authors_elem.find_all('li')] if authors_elem else ["No authors found"]
            
            date_elem = article.find('time')
            date = date_elem['datetime'] if date_elem and 'datetime' in date_elem.attrs else "No date found"
            
            # Extract year from date for filtering
            article_year = None
            if date and date != "No date found":
                try:
                    article_year = int(date[:4])  # Extract year from YYYY-MM-DD format
                except (ValueError, IndexError):
                    pass

            image_elem = article.find('img')
            image_url = image_elem['src'] if image_elem and 'src' in image_elem.attrs else "No image found"

            # When target_year is specified, the website should already be filtering by year
            # So we include all articles from the page (they should all be from the target year)
            # When target_year is None, we include all articles
            articles.append({
                'title': title,
                'url': url,
                'description': description,
                'authors': authors,
                'date': date,               
                'year': article_year,
                'image_url': image_url
            })

        return articles
    def run(self, num_articles=None, max_pages=None, year=None, count_only=True):
        """
        Runs the tool to fetch news articles from Nature's website.

        Parameters:
            num_articles (int or None): The number of articles to fetch. Set to None to fetch all available articles.
            max_pages (int or None): The maximum number of pages to fetch. Set to None to fetch all available pages.
            year (int, optional): Filter articles by specific year.
            count_only (bool): If True, return only the count instead of full details.

        Returns:
            list or dict: A list of dictionaries containing article information, or a dictionary with counts.
        """
        return self.search_new(num_articles, max_pages, year, count_only)
    
    def search_new(self, num_articles=None, max_pages=None, year=None, count_only=True):
        """
        Fetches news articles from Nature's website with optional year filtering and counting.

        Parameters:
            num_articles (int or None): The number of articles to fetch. Set to None to fetch all available articles.
            max_pages (int or None): The maximum number of pages to fetch. Set to None to fetch all available pages.
            year (int, optional): Filter articles by specific year.
            count_only (bool): If True, return only the count instead of full details.

        Returns:
            list or dict: A list of dictionaries containing article information, or a dictionary with counts.
        """
        all_articles = []
        page_number = 1
        year_counts = {} if count_only else None

        try:
            # Check if we should fetch all articles/pages (None values) or limit by specified values
            should_continue = lambda: (num_articles is None or len(all_articles) < num_articles) and (max_pages is None or page_number <= max_pages)
            
            while should_continue():
                html_content = self.fetch_page(page_number, year)
                
                # If fetch_page returns None, we've reached the end of available pages
                if html_content is None:
                    break
                
                page_articles = self.parse_articles(html_content, year)
                
                if not page_articles:
                    break  # No more articles found

                # Debug: Print how many articles found on this page
                print(f"Page {page_number}: Found {len(page_articles)} articles")

                if count_only:
                    # Count articles by year
                    for article in page_articles:
                        article_year = article.get('year')
                        if article_year:
                            year_counts[article_year] = year_counts.get(article_year, 0) + 1
                else:
                    all_articles.extend(page_articles)
                
                page_number += 1
                time.sleep(0.2)  # Reduced sleep time for faster counting

            if count_only:
                if year:
                    # Return count for specific year
                    return {
                        "result": {
                            "year": year,
                            "article_count": year_counts.get(year, 0),
                            "pages_sampled": page_number - 1,
                            "message": f"Found {year_counts.get(year, 0)} articles from {year} (sampled from {page_number-1} pages)",
                        },
                        "success": True
                    }
                else:
                    # Return counts for all years found
                    return {
                        "result": {
                            "year_counts": year_counts,
                            "total_articles": sum(year_counts.values()),
                            "pages_sampled": page_number - 1,
                            "message": f"Found articles across {len(year_counts)} years: {year_counts} (sampled from {page_number-1} pages)",
                        },
                        "success": True
                    }
            else:
                # If num_articles is None, return all articles; otherwise limit to num_articles
                return {"result": all_articles if num_articles is None else all_articles[:num_articles], "success": True}
        except Exception as e:
            return {"error": f"Error fetching articles: {str(e)}", "details": "This could be due to network issues, website changes, or invalid parameters.", "success": False, "traceback": traceback.format_exc()}

if __name__ == "__main__":
    tool = Nature_News_Fetcher_Tool()
    tool.embed_tool()