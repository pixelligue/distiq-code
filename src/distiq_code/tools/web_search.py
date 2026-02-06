"""
Web Search Tools

Multiple search providers for web search and content extraction:

1. Jina AI (Primary):
   - s.jina.ai — Web search (returns URLs + content)
   - r.jina.ai — URL to Markdown reader (FREE, no key needed!)
   - 1M free tokens on signup

2. Tavily (Alternative):
   - Built for AI agents
   - 1000 free searches/month
   - Clean, structured results

Features:
- Automatic fallback between providers
- Clean LLM-ready Markdown output
- No heavy dependencies
"""

import os
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger


@dataclass
class SearchResult:
    """Single search result."""
    
    title: str
    url: str
    snippet: str
    content: str | None = None  # Full markdown content if fetched


@dataclass 
class SearchResponse:
    """Web search response."""
    
    query: str
    results: list[SearchResult]
    total_results: int
    
    def to_context(self, max_results: int = 3) -> str:
        """Format as context for LLM."""
        parts = [f"## Web Search Results for: {self.query}\n"]
        
        for i, result in enumerate(self.results[:max_results], 1):
            parts.append(f"### {i}. {result.title}")
            parts.append(f"URL: {result.url}")
            
            if result.content:
                # Truncate content if too long
                content = result.content[:2000]
                if len(result.content) > 2000:
                    content += "\n... (truncated)"
                parts.append(f"\n{content}")
            else:
                parts.append(f"\n{result.snippet}")
            
            parts.append("")
        
        return "\n".join(parts)


class JinaSearchClient:
    """
    Jina AI Search and Reader client.
    
    Usage:
        client = JinaSearchClient()
        
        # Search the web
        results = await client.search("Python async best practices")
        
        # Read a specific URL
        content = await client.read_url("https://docs.python.org")
    """
    
    SEARCH_URL = "https://s.jina.ai/"
    READER_URL = "https://r.jina.ai/"
    
    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize Jina client.
        
        Args:
            api_key: Jina AI API key (or JINA_API_KEY env var)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {
                "Accept": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self._client = httpx.AsyncClient(
                headers=headers,
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client
    
    async def search(
        self,
        query: str,
        num_results: int = 5,
        fetch_content: bool = True,
    ) -> SearchResponse:
        """
        Search the web using Jina Search API.
        
        Args:
            query: Search query
            num_results: Number of results (max 5 for free tier)
            fetch_content: Whether to include full page content
            
        Returns:
            SearchResponse with results
        """
        client = await self._get_client()
        
        # Build URL
        url = f"{self.SEARCH_URL}"
        params = {"q": query}
        
        logger.debug(f"Searching: {query}")
        
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            results = []
            for item in data.get("data", [])[:num_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    content=item.get("content") if fetch_content else None,
                ))
            
            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Search failed: {e.response.status_code}")
            # Return empty results on error
            return SearchResponse(query=query, results=[], total_results=0)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return SearchResponse(query=query, results=[], total_results=0)
    
    async def read_url(
        self,
        url: str,
        timeout: float | None = None,
    ) -> str:
        """
        Read a URL and convert to Markdown.
        
        Args:
            url: URL to read
            timeout: Optional custom timeout
            
        Returns:
            Markdown content of the page
        """
        client = await self._get_client()
        
        # Jina Reader: prepend r.jina.ai to URL
        reader_url = f"{self.READER_URL}{url}"
        
        logger.debug(f"Reading URL: {url}")
        
        try:
            response = await client.get(
                reader_url,
                timeout=timeout or self.timeout,
            )
            response.raise_for_status()
            
            return response.text
            
        except Exception as e:
            logger.error(f"Read URL failed: {e}")
            return f"Error reading URL: {e}"
    
    async def search_and_read(
        self,
        query: str,
        num_results: int = 3,
    ) -> SearchResponse:
        """
        Search and fetch full content for each result.
        
        Slower but provides complete page content.
        """
        # First search
        results = await self.search(query, num_results, fetch_content=True)
        
        # If search already returned content, we're done
        if results.results and results.results[0].content:
            return results
        
        # Otherwise fetch content for each URL
        for result in results.results:
            if not result.content:
                result.content = await self.read_url(result.url)
        
        return results
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Synchronous wrapper for non-async contexts
class JinaSearch:
    """Synchronous Jina Search wrapper."""
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
    
    def search(self, query: str, num_results: int = 5) -> SearchResponse:
        """Synchronous search."""
        import asyncio
        
        async def _search():
            client = JinaSearchClient(self.api_key)
            try:
                return await client.search(query, num_results)
            finally:
                await client.close()
        
        return asyncio.run(_search())
    
    def read_url(self, url: str) -> str:
        """Synchronous URL read."""
        import asyncio
        
        async def _read():
            client = JinaSearchClient(self.api_key)
            try:
                return await client.read_url(url)
            finally:
                await client.close()
        
        return asyncio.run(_read())


# Global client instance
_client: JinaSearchClient | None = None


def get_jina_client() -> JinaSearchClient:
    """Get global Jina client."""
    global _client
    if _client is None:
        _client = JinaSearchClient()
    return _client


async def web_search(
    query: str,
    num_results: int = 5,
) -> SearchResponse:
    """
    Quick web search.
    
    Args:
        query: Search query
        num_results: Number of results
        
    Returns:
        SearchResponse
    """
    client = get_jina_client()
    return await client.search(query, num_results)


async def read_url(url: str) -> str:
    """
    Quick URL read.
    
    Args:
        url: URL to read
        
    Returns:
        Markdown content
    """
    client = get_jina_client()
    return await client.read_url(url)


# =============================================================================
# Tavily Search Client
# =============================================================================

class TavilySearchClient:
    """
    Tavily search client - built for AI agents.
    
    Features:
    - 1000 free searches/month
    - Clean, structured results
    - Built-in content extraction
    
    Usage:
        client = TavilySearchClient()
        results = await client.search("Python best practices")
    """
    
    API_URL = "https://api.tavily.com/search"
    
    def __init__(self, api_key: str | None = None):
        """
        Initialize Tavily client.
        
        Args:
            api_key: Tavily API key (or TAVILY_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client
    
    async def search(
        self,
        query: str,
        num_results: int = 5,
        include_answer: bool = False,
    ) -> SearchResponse:
        """
        Search using Tavily API.
        
        Args:
            query: Search query
            num_results: Number of results (1-10)
            include_answer: Include AI-generated answer
            
        Returns:
            SearchResponse
        """
        if not self.api_key:
            logger.warning("TAVILY_API_KEY not set")
            return SearchResponse(query=query, results=[], total_results=0)
        
        client = await self._get_client()
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": min(num_results, 10),
            "include_answer": include_answer,
            "include_raw_content": False,
        }
        
        logger.debug(f"Tavily search: {query}")
        
        try:
            response = await client.post(self.API_URL, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            for item in data.get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    content=item.get("raw_content"),
                ))
            
            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Tavily search failed: {e.response.status_code}")
            return SearchResponse(query=query, results=[], total_results=0)
        except Exception as e:
            logger.error(f"Tavily error: {e}")
            return SearchResponse(query=query, results=[], total_results=0)
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# Unified Search with Fallback
# =============================================================================

_tavily_client: TavilySearchClient | None = None


def get_tavily_client() -> TavilySearchClient:
    """Get global Tavily client."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilySearchClient()
    return _tavily_client


async def smart_search(
    query: str,
    num_results: int = 5,
    prefer: str = "jina",
) -> SearchResponse:
    """
    Smart search with automatic fallback.
    
    Tries providers in order:
    1. Jina (if JINA_API_KEY set)
    2. Tavily (if TAVILY_API_KEY set)
    3. Return empty if both fail
    
    Args:
        query: Search query
        num_results: Number of results
        prefer: Preferred provider ("jina" or "tavily")
        
    Returns:
        SearchResponse from first successful provider
    """
    providers = []
    
    # Order providers by preference
    if prefer == "tavily":
        if os.getenv("TAVILY_API_KEY"):
            providers.append(("tavily", get_tavily_client()))
        if os.getenv("JINA_API_KEY"):
            providers.append(("jina", get_jina_client()))
    else:
        if os.getenv("JINA_API_KEY"):
            providers.append(("jina", get_jina_client()))
        if os.getenv("TAVILY_API_KEY"):
            providers.append(("tavily", get_tavily_client()))
    
    # Try each provider
    for name, client in providers:
        logger.debug(f"Trying {name} search...")
        
        results = await client.search(query, num_results)
        
        if results.results:
            logger.debug(f"{name} returned {len(results.results)} results")
            return results
    
    # No providers available or all failed
    logger.warning("No search providers available. Set JINA_API_KEY or TAVILY_API_KEY")
    return SearchResponse(query=query, results=[], total_results=0)
