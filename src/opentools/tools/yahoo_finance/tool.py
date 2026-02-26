# source code: https://github.com/inclusionAI/AWorld/blob/main/examples/gaia/mcp_collections/tools/yahoofinance.py
import os, time, traceback, sys
from datetime import datetime, timezone
from typing import Any

import yfinance as yf
from pathlib import Path
from typing import Any, Literal, Optional
import re
import requests
import logging
import io
import contextlib
from datetime import timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..')))
from opentools.core.base import BaseTool
from pydantic import Field, BaseModel
from pydantic.fields import FieldInfo

class YFinanceMetadata(BaseModel):
    """Metadata for Yahoo Finance operation results."""

    symbol: str
    operation: str
    execution_time: float | None = None
    data_points: int | None = None
    error_type: str | None = None
    timestamp: str | None = None


class Yahoo_Finance_Tool(BaseTool):
    """Yahoo_Finance_Tool
    ---------------------
    Purpose:
        A comprehensive Yahoo Finance data retrieval tool that provides real-time stock quotes, historical data, company information, financial statements, and market data. Supports multiple data intervals, flexible date ranges.

    Core Capabilities:
        - Real-time stock quote retrieval with current price, volume, market cap, and key metrics
        - Historical OHLCV data with flexible intervals and date ranges
        - Company information including sector, industry, employees, and business details
        - Financial statements (income statements, balance sheets, cash flow statements) in annual/quarterly formats

    Intended Use:
        Use this tool when you need to retrieve real-time stock quotes, historical data, company information, financial statements, and market data.

    Limitations:
        - Data availability depends on Yahoo Finance API, some symbols may have limited data
        - Historical data limited by Yahoo Finance availability
        - Financial statements may not be available for all companies
        - Real-time data may have slight delays
        - Rate limiting may apply for frequent requests
        - Some international markets may have limited data
    """

    def __init__(self) -> None:
        super().__init__(
            type='function',
            name="Yahoo_Finance_Tool",
            description="""A comprehensive Yahoo Finance data retrieval tool that provides real-time stock quotes, historical data, company information, financial statements, and market data. Supports multiple data intervals, flexible date ranges. CAPABILITIES: Real-time stock quote retrieval with current price, volume, market cap, and key metrics, historical OHLCV data with flexible intervals and date ranges, company information including sector, industry, employees, and business details, financial statements (income statements, balance sheets, cash flow statements) in annual/quarterly formats. SYNONYMS: Yahoo Finance tool, stock data extractor, financial data retriever, stock quote tool, historical data fetcher, company information tool, financial statement analyzer, market data tool, stock analysis tool, financial data API. EXAMPLES: 'Get current stock quote for Apple Inc.', 'Get historical daily data for Microsoft from 2024', 'Get comprehensive company information for Google', 'Get annual income statement for Apple Inc.'.""",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform: 'get_stock_quote', 'get_historical_data', 'get_company_info', 'get_financial_statements', or 'get_market_summary'",
                        "enum": ["get_stock_quote", "get_historical_data", "get_company_info", "get_financial_statements", "get_market_summary"]
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)."
                    },
                    "start_date": {
                        "type": ["string", "null"],
                        "description": "Start date for historical data (YYYY-MM-DD format). Required for 'get_historical_data' operation."
                    },
                    "end_date": {
                        "type": ["string", "null"],
                        "description": "End date for historical data (YYYY-MM-DD format). Required for 'get_historical_data' operation."
                    },
                    "interval": {
                        "type": "string",
                        "description": "Data interval for historical data (1d, 1wk, 1mo, etc., default: 1y)"
                    },
                    "statement_type": {
                        "type": ["string", "null"],
                        "description": "Type of financial statement (income_statement, balance_sheet, cash_flow). Required for 'get_financial_statements' operation."
                    },
                    "period_type": {
                        "type": "string",
                        "description": "Period type for financial statements (annual or quarterly, default: annual)"
                    },
                    "max_rows_preview": {
                        "type": "integer",
                        "description": "Maximum rows to show in historical data preview (0 for all, default: 0)"
                    },
                    "max_columns_preview": {
                        "type": "integer",
                        "description": "Maximum periods to show in financial statements (0 for all, default: 4)"
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format: 'markdown', 'json', or 'text' (default: json)",
                        "enum": ["markdown", "json", "text"]
                    }
                },
                "required": ["operation", "symbol"],
                "additionalProperties": False,
            },
            strict=False,
            category="financial_data",
            tags=["yahoo_finance", "stock_data", "financial_data", "stock_quotes", "historical_data", "company_info", "financial_statements", "market_data", "yfinance", "investment_tools", "financial_analysis"],
            limitation="Data availability depends on Yahoo Finance API, some symbols may have limited data, historical data limited by Yahoo Finance availability, financial statements may not be available for all companies, real-time data may have slight delays, rate limiting may apply for frequent requests, some international markets may have limited data",
            agent_type="Search-Agent",
            demo_commands= {
                "command": "reponse = tool.run(operation='get_stock_quote', symbol='AAPL')",
                "description": "Get current stock quote for Apple Inc."
            }
        )
    def get_metadata(self):
        return super().get_metadata()
    
    def embed_tool(self):
        return super().embed_tool()

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize user-provided symbols into a yfinance-friendly ticker.

        Common failure mode in GAIA runs: the agent passes Google Finance-style symbols
        like "AAPL:NASDAQ". yfinance expects just "AAPL" (for US listings), and passing
        the colon form can lead to yfinance failing to resolve timezone/metadata.
        """
        s = (symbol or "").strip()
        # Remove leading '$' or other common UI prefixes.
        s = s.lstrip("$").strip()
        # Collapse internal whitespace
        s = re.sub(r"\s+", "", s)

        # Convert Google Finance "TICKER:EXCHANGE" -> "TICKER" (best-effort).
        # This is intentionally conservative; we avoid attempting complex exchange mappings.
        if ":" in s:
            left, _right = s.split(":", 1)
            if left:
                s = left

        return s.upper()

    @contextlib.contextmanager
    def _quiet_yfinance(self):
        """
        Suppress yfinance noisy logging/prints.

        yfinance sometimes emits lines like:
        - "ERROR AAPL: No timezone found, symbol may be delisted"
        - "Failed to get ticker ... Expecting value: line 1 column 1 (char 0)"
        even when we successfully recover via the HTTP fallback. These logs confuse users/agents.
        """
        buf = io.StringIO()
        ylog = logging.getLogger("yfinance")
        prev_level = ylog.level
        prev_propagate = ylog.propagate
        try:
            ylog.setLevel(logging.CRITICAL)
            ylog.propagate = False
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                yield
        finally:
            ylog.setLevel(prev_level)
            ylog.propagate = prev_propagate

    def _is_timezone_error(self, err: BaseException) -> bool:
        msg = str(err).lower()
        return (
            "no timezone found" in msg
            or "no time zone found" in msg
            or "timezone" in msg and "not found" in msg
        )

    def _is_yfinance_json_decode_symptom(self, err: BaseException) -> bool:
        msg = str(err).lower()
        return (
            "expecting value" in msg
            and "line 1 column 1" in msg
            and "char 0" in msg
        )

    def _normalize_interval(self, interval: str) -> tuple[str, bool]:
        """
        Normalize intervals into values supported by yfinance/Yahoo chart API.

        yfinance does NOT support a true "1y" interval. We treat user-provided
        "1y"/"1yr"/"1year" as a request for *yearly aggregated* results, fetched
        at monthly granularity and then reduced.
        """
        itv = (interval or "").strip().lower()
        if itv in {"1y", "1yr", "1year", "1yearly", "year", "yearly"}:
            return "1mo", True
        return interval, False

    def _reduce_to_yearly_max_close(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Reduce OHLCV records into one record per year using max Close for that year.

        Output format keeps a "Date" field so existing formatting continues to work.
        """
        per_year: dict[int, dict[str, Any]] = {}
        for r in records:
            d = r.get("Date") or r.get("Datetime")
            if not d:
                continue
            # Date could be "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS" or "YYYY"
            try:
                year = int(str(d)[:4])
            except Exception:
                continue

            close = r.get("Close")
            if close is None:
                continue

            entry = per_year.get(year)
            if entry is None:
                per_year[year] = {"Date": str(year), "Close": close}
            else:
                if close > entry["Close"]:
                    entry["Close"] = close

        return [per_year[y] for y in sorted(per_year)]

    def _format_historical_records_full(self, records: list[dict[str, Any]]) -> str:
        """
        Render all historical records (no preview). Intended for max_rows_preview=0.
        """
        lines: list[str] = [f"# Historical Data ({len(records)} records)", "", "**Data:**"]
        for r in records:
            date = r.get("Date") or r.get("Datetime") or "N/A"
            close = r.get("Close")
            open_ = r.get("Open")
            high = r.get("High")
            low = r.get("Low")
            vol = r.get("Volume")

            parts: list[str] = []
            if isinstance(close, (int, float)):
                parts.append(f"Close ${close:.2f}")
            elif close is not None:
                parts.append(f"Close {close}")

            if isinstance(open_, (int, float)):
                parts.append(f"Open ${open_:.2f}")
            if isinstance(high, (int, float)):
                parts.append(f"High ${high:.2f}")
            if isinstance(low, (int, float)):
                parts.append(f"Low ${low:.2f}")
            if isinstance(vol, (int, float)):
                parts.append(f"Volume {int(vol):,}")
            elif vol is not None:
                parts.append(f"Volume {vol}")

            tail = ", ".join(parts) if parts else "No fields"
            lines.append(f"- {date}: {tail}")

        return "\n".join(lines)

    def _parse_yyyy_mm_dd(self, s: str) -> datetime:
        # Parse date-only input (YYYY-MM-DD) as UTC midnight for comparisons.
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)

    def _add_years(self, dt: datetime, years: int) -> datetime:
        """
        Add years to a datetime safely (handles Feb 29 by clamping to Feb 28).
        """
        try:
            return dt.replace(year=dt.year + years)
        except ValueError:
            # Feb 29 -> Feb 28
            return dt.replace(month=2, day=28, year=dt.year + years)

    def _fetch_chart_history_via_http(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str,
    ) -> list[dict[str, Any]]:
        """
        Fallback historical data fetcher using Yahoo's public chart JSON endpoint.

        This avoids some yfinance failure modes where metadata/timezone fetch fails
        (e.g., JSON decode errors due to blocked/empty responses), which leads to
        "No timezone found" and empty history frames even for valid tickers like AAPL.
        """
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        start_dt = self._parse_yyyy_mm_dd(start)
        end_dt = self._parse_yyyy_mm_dd(end)

        def _fetch_once(p1: int, p2: int) -> list[dict[str, Any]]:
            params = {
                "period1": str(p1),
                "period2": str(p2),
                "interval": interval,
                "includePrePost": "false",
                "includeAdjustedClose": "true",
                "events": "div,splits",
                "corsDomain": "finance.yahoo.com",
            }
            resp = requests.get(
                url,
                params=params,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=20,
            )
            if resp.status_code != 200:
                raise ValueError(f"Yahoo chart HTTP {resp.status_code} for {symbol}")

            try:
                payload = resp.json()
            except Exception as e:
                raise ValueError(f"Yahoo chart JSON decode failed for {symbol}: {e}") from e

            chart = (payload or {}).get("chart") or {}
            if chart.get("error"):
                raise ValueError(f"Yahoo chart API error for {symbol}: {chart['error']}")

            results = chart.get("result") or []
            if not results:
                return []

            r0 = results[0] or {}
            timestamps = r0.get("timestamp") or []
            indicators = (r0.get("indicators") or {}).get("quote") or []
            quote0 = indicators[0] if indicators else {}

            opens = quote0.get("open") or []
            highs = quote0.get("high") or []
            lows = quote0.get("low") or []
            closes = quote0.get("close") or []
            volumes = quote0.get("volume") or []

            def _get(arr: list[Any], idx: int) -> Any:
                try:
                    return arr[idx]
                except Exception:
                    return None

            out: list[dict[str, Any]] = []
            for i, ts in enumerate(timestamps):
                if ts is None:
                    continue
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                # We filter by the caller's requested bounds later; keep all here.
                row = {
                    "Date": dt.date().isoformat(),
                    "Open": _get(opens, i),
                    "High": _get(highs, i),
                    "Low": _get(lows, i),
                    "Close": _get(closes, i),
                    "Volume": _get(volumes, i),
                }
                row = {k: v for k, v in row.items() if v is not None or k == "Date"}
                out.append(row)
            return out

        # Yahoo chart API can truncate very long "1mo" ranges due to response size limits.
        # For long spans, chunk by years to preserve the earliest data.
        chunk_years = 15 if interval in {"1mo"} else 0
        all_rows: list[dict[str, Any]] = []
        if chunk_years:
            cur = start_dt
            while cur <= end_dt:
                nxt = self._add_years(cur, chunk_years)
                if nxt > end_dt:
                    nxt = end_dt
                p1 = int(cur.timestamp())
                p2 = int(nxt.timestamp()) + 86400
                all_rows.extend(_fetch_once(p1, p2))
                # move forward one day to avoid overlapping the boundary too much
                cur = nxt.replace(tzinfo=timezone.utc) + timedelta(days=1)
        else:
            p1 = int(start_dt.timestamp())
            p2 = int(end_dt.timestamp()) + 86400
            all_rows = _fetch_once(p1, p2)

        # Filter + dedupe by Date.
        by_date: dict[str, dict[str, Any]] = {}
        for r in all_rows:
            d = r.get("Date")
            if not d:
                continue
            try:
                dt = self._parse_yyyy_mm_dd(str(d))
            except Exception:
                continue
            if dt < start_dt or dt > end_dt:
                continue
            # last write wins (later chunks overwrite earlier duplicates)
            by_date[str(d)] = r

        return [by_date[d] for d in sorted(by_date)]
    
    def _format_financial_data(self, data: Any, data_type: str) -> str:
        """Format financial data for LLM consumption.

        Args:
            data: Raw financial data
            data_type: Type of data for context

        Returns:
            LLM-friendly formatted string
        """
        if isinstance(data, dict):
            if data_type == "quote":
                return self._format_quote_data(data)
            elif data_type == "company":
                return self._format_company_data(data)
        elif isinstance(data, list):
            if data_type == "historical":
                return self._format_historical_data(data)
            elif data_type == "market_summary":
                return self._format_market_summary_data(data)
            elif data_type == "news":
                return self._format_news_list_data(data)

        return str(data)

    def _format_quote_data(self, quote: dict[str, Any]) -> str:
        """Format stock quote data for LLM."""
        lines = [f"# Stock Quote: {quote.get('symbol', 'N/A')}"]

        if quote.get("companyName"):
            lines.append(f"**Company:** {quote['companyName']}")

        if quote.get("currentPrice"):
            lines.append(f"**Current Price:** ${quote['currentPrice']:.2f} {quote.get('currency', '')}")

        if quote.get("previousClose"):
            change = quote.get("currentPrice", 0) - quote.get("previousClose", 0)
            change_pct = (change / quote["previousClose"]) * 100 if quote.get("previousClose") else 0
            direction = "📈" if change >= 0 else "📉"
            lines.append(f"**Change:** {direction} ${change:.2f} ({change_pct:.2f}%)")

        if quote.get("dayHigh") and quote.get("dayLow"):
            lines.append(f"**Day Range:** ${quote['dayLow']:.2f} - ${quote['dayHigh']:.2f}")

        if quote.get("volume"):
            lines.append(f"**Volume:** {quote['volume']:,}")

        if quote.get("marketCap"):
            lines.append(f"**Market Cap:** ${quote['marketCap']:,}")

        return "\n".join(lines)

    def _format_company_data(self, company: dict[str, Any]) -> str:
        """Format company information for LLM."""
        lines = [f"# Company Information: {company.get('symbol', 'N/A')}"]

        if company.get("longName"):
            lines.append(f"**Company Name:** {company['longName']}")

        if company.get("sector"):
            lines.append(f"**Sector:** {company['sector']}")

        if company.get("industry"):
            lines.append(f"**Industry:** {company['industry']}")

        if company.get("fullTimeEmployees"):
            lines.append(f"**Employees:** {company['fullTimeEmployees']:,}")

        if company.get("city") and company.get("country"):
            location = f"{company['city']}, {company['country']}"
            if company.get("state"):
                location = f"{company['city']}, {company['state']}, {company['country']}"
            lines.append(f"**Location:** {location}")

        if company.get("website"):
            lines.append(f"**Website:** {company['website']}")

        if company.get("longBusinessSummary"):
            summary = (
                company["longBusinessSummary"][:500] + "..."
                if len(company["longBusinessSummary"]) > 500
                else company["longBusinessSummary"]
            )
            lines.extend(["\n**Business Summary:**", summary])

        return "\n".join(lines)

    def _format_historical_data(self, data: list[dict[str, Any]]) -> str:
        """Format historical data for LLM."""
        if not data:
            return "No historical data available."

        lines = [f"# Historical Data ({len(data)} records)"]

        # Show first few and last few records
        preview_count = min(3, len(data))

        lines.append("\n**Recent Data:**")
        for record in data[-preview_count:]:
            date = record.get("Date", record.get("Datetime", "N/A"))
            close = record.get("Close", 0)
            volume = record.get("Volume", 0)
            lines.append(f"- {date}: Close ${close:.2f}, Volume {volume:,}")

        if len(data) > preview_count * 2:
            lines.append(f"\n... {len(data) - preview_count * 2} more records ...")

        if len(data) > preview_count:
            lines.append("\n**Earliest Data:**")
            for record in data[:preview_count]:
                date = record.get("Date", record.get("Datetime", "N/A"))
                close = record.get("Close", 0)
                volume = record.get("Volume", 0)
                lines.append(f"- {date}: Close ${close:.2f}, Volume {volume:,}")

        return "\n".join(lines)

    def _format_news_list_data(self, news_list: list[dict[str, Any]]) -> str:
        """Format news list for LLM."""
        if not news_list:
            return "No news articles found."

        lines = [f"# Financial News ({len(news_list)} articles)"]

        for i, article in enumerate(news_list, 1):
            lines.append(f"\n## {i}. {article.get('title', 'No Title')}")
            if article.get("publisher"):
                lines.append(f"**Publisher:** {article['publisher']}")
            if article.get("providerPublishTime"):
                lines.append(f"**Published:** {article['providerPublishTime']}")
            if article.get("link"):
                lines.append(f"**Link:** {article['link']}")

        return "\n".join(lines)

    def _format_market_summary_data(self, summaries: list[dict[str, Any]]) -> str:
        """Format market summary for LLM."""
        if not summaries:
            return "No market data available."

        lines = ["# Market Summary"]

        for summary in summaries:
            symbol = summary.get("symbol", "N/A")
            name = summary.get("name", symbol)
            price = summary.get("currentPrice", 0)
            change = summary.get("change", 0)
            change_pct = summary.get("percentChange", 0)

            direction = "📈" if change >= 0 else "📉"
            lines.append(f"\n**{name} ({symbol})**")
            lines.append(f"Price: ${price:.2f} {direction} {change:+.2f} ({change_pct:+.2f}%)")

        return "\n".join(lines)

    def run(
        self,
        operation: str = Field(description="The operation to perform: 'get_stock_quote', 'get_historical_data', 'get_company_info', 'get_financial_statements', or 'get_market_summary'"),
        symbol: str | None = Field(default=None, description="Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"),
        start_date: str | None = Field(default=None, description="Start date for historical data (YYYY-MM-DD format)"),
        end_date: str | None = Field(default=None, description="End date for historical data (YYYY-MM-DD format)"),
        interval: str = Field(default="1y", description="Data interval for historical data (1d, 1wk, 1mo, etc.)"),
        statement_type: str | None = Field(default=None, description="Type of financial statement (income_statement, balance_sheet, cash_flow)"),
        period_type: str = Field(default="annual", description="Period type for financial statements (annual or quarterly)"),
        max_rows_preview: int = Field(default=0, description="Maximum rows to show in historical data preview (0 for all)"),
        max_columns_preview: int = Field(default=4, description="Maximum periods to show in financial statements (0 for all)"),
        output_format: str = Field(default="json", description="Output format: 'markdown', 'json', or 'text'"),
    ):
        """Unified function for Yahoo Finance operations.
        
        This function handles all Yahoo Finance operations based on the operation parameter.
        
        Args:
            operation: The operation to perform
            symbol: Stock ticker symbol
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD)
            interval: Data interval for historical data
            statement_type: Type of financial statement
            period_type: Period type for financial statements
            max_rows_preview: Maximum rows for historical data preview
            max_columns_preview: Maximum periods for financial statements
            output_format: Format for the response
            
        Returns:
            Dictionary with operation results and metadata
        """
        # Handle FieldInfo objects
        if isinstance(operation, FieldInfo):
            operation = operation.default
        if isinstance(symbol, FieldInfo):
            symbol = symbol.default
        if isinstance(start_date, FieldInfo):
            start_date = start_date.default
        if isinstance(end_date, FieldInfo):
            end_date = end_date.default
        if isinstance(interval, FieldInfo):
            interval = interval.default
        if isinstance(statement_type, FieldInfo):
            statement_type = statement_type.default
        if isinstance(period_type, FieldInfo):
            period_type = period_type.default
        if isinstance(max_rows_preview, FieldInfo):
            max_rows_preview = max_rows_preview.default
        if isinstance(max_columns_preview, FieldInfo):
            max_columns_preview = max_columns_preview.default
        if isinstance(output_format, FieldInfo):
            output_format = output_format.default

        start_time = time.time()

        try:
            if operation == "get_stock_quote":
                if not symbol:
                    return {
                        "error": "Error: 'symbol' parameter is REQUIRED for 'get_stock_quote' operation. Example: tool.run(operation='get_stock_quote', symbol='AAPL')",
                        "metadata": YFinanceMetadata(
                            symbol=symbol or "",
                            operation="get_stock_quote",
                            execution_time=time.time() - start_time,
                            error_type="missing_symbol",
                        ).model_dump(),
                        "success": False
                    }
                return self.mcp_get_stock_quote(symbol)

            elif operation == "get_historical_data":
                if not symbol:
                    return {
                        "error": "Error: 'symbol' parameter is REQUIRED for 'get_historical_data' operation. Example: tool.run(operation='get_historical_data', symbol='AAPL', start_date='2024-01-01', end_date='2024-12-31')",
                        "metadata": YFinanceMetadata(
                            symbol=symbol or "",
                            operation="get_historical_data",
                            execution_time=time.time() - start_time,
                            error_type="missing_symbol",
                        ).model_dump(),
                        "success": False
                    }
                if not start_date:
                    return {
                        "error": "Error: 'start_date' parameter is REQUIRED for 'get_historical_data' operation. Example: tool.run(operation='get_historical_data', symbol='AAPL', start_date='2024-01-01', end_date='2024-12-31')",
                        "metadata": YFinanceMetadata(
                            symbol=symbol or "",
                            operation="get_historical_data",
                            execution_time=time.time() - start_time,
                            error_type="missing_start_date",
                        ).model_dump(),
                        "success": False
                    }
                if not end_date:
                    return {
                        "error": "Error: 'end_date' parameter is REQUIRED for 'get_historical_data' operation. Example: tool.run(operation='get_historical_data', symbol='AAPL', start_date='2024-01-01', end_date='2024-12-31')",
                        "metadata": YFinanceMetadata(
                            symbol=symbol or "",
                            operation="get_historical_data",
                            execution_time=time.time() - start_time,
                            error_type="missing_end_date",
                        ).model_dump(),
                        "success": False
                    }
                return self.mcp_get_historical_data(symbol, start_date, end_date, interval, max_rows_preview)

            elif operation == "get_company_info":
                if not symbol:
                    return {
                        "error": "Error: 'symbol' parameter is REQUIRED for 'get_company_info' operation. Example: tool.run(operation='get_company_info', symbol='AAPL')",
                        "metadata": YFinanceMetadata(
                            symbol=symbol or "",
                            operation="get_company_info",
                            execution_time=time.time() - start_time,
                            error_type="missing_symbol",
                        ).model_dump(),
                        "success": False
                    }
                return self.mcp_get_company_info(symbol)

            elif operation == "get_financial_statements":
                if not symbol:
                    return {
                        "error": "Error: 'symbol' parameter is REQUIRED for 'get_financial_statements' operation. Example: tool.run(operation='get_financial_statements', symbol='AAPL', statement_type='income_statement')",
                        "metadata": YFinanceMetadata(
                            symbol=symbol or "",
                            operation="get_financial_statements",
                            execution_time=time.time() - start_time,
                            error_type="missing_symbol",
                        ).model_dump(),
                        "success": False
                    }
                if not statement_type:
                    return {
                        "error": "Error: 'statement_type' parameter is REQUIRED for 'get_financial_statements' operation. Example: tool.run(operation='get_financial_statements', symbol='AAPL', statement_type='income_statement')",
                        "metadata": YFinanceMetadata(
                            symbol=symbol or "",
                            operation="get_financial_statements",
                            execution_time=time.time() - start_time,
                            error_type="missing_statement_type",
                        ).model_dump(),
                        "success": False
                    }
                return self.mcp_get_financial_statements(symbol, statement_type, period_type, max_columns_preview)

            else:
                return {
                    "error": f"Error: Unknown operation '{operation}'. Supported operations: 'get_stock_quote', 'get_historical_data', 'get_company_info', 'get_financial_statements'",
                    "metadata": YFinanceMetadata(
                        symbol=symbol or "",
                        operation=operation or "unknown",
                        execution_time=time.time() - start_time,
                        error_type="unknown_operation",
                    ).model_dump(),
                    "success": False
                }

        except Exception as e:
            error_msg = f"Yahoo Finance operation failed: {str(e)}"
            print(f"Error in run: {traceback.format_exc()}")
            return {
                "error": error_msg,
                "metadata": YFinanceMetadata(
                    symbol=symbol or "",
                    operation=operation or "unknown",
                    execution_time=time.time() - start_time,
                    error_type=type(e).__name__,
                ).model_dump(),
                "success": False,
                "traceback": traceback.format_exc()
            }

    def mcp_get_stock_quote(
        self,
        symbol: str = Field(description="Stock ticker symbol (e.g., AAPL, MSFT)"),
    ) :
        """Get current stock quote information.

        Fetches real-time stock quote data including current price, daily changes,
        volume, market cap, and other key metrics for the specified ticker symbol.

        Args:
            symbol: The stock ticker symbol to fetch quote for

        Returns:
            ActionResponse with formatted quote data and metadata
        """
        # Handle FieldInfo objects
        if isinstance(symbol, FieldInfo):
            symbol = symbol.default

        try:
            start_time = time.time()

            symbol = self._normalize_symbol(symbol)
            with self._quiet_yfinance():
                ticker = yf.Ticker(symbol)
                info = ticker.info

            if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
                # Try to get basic history to validate symbol
                with self._quiet_yfinance():
                    hist = ticker.history(period="1d")
                if hist.empty:
                    raise ValueError(f"No data found for symbol: {symbol}. It might be invalid or delisted.")
                raise ValueError(f"Could not retrieve detailed quote for symbol: {symbol}. Limited data available.")

            # Extract key quote information
            quote_data = {
                "symbol": symbol.upper(),
                "companyName": info.get("shortName", info.get("longName")),
                "currentPrice": info.get("regularMarketPrice", info.get("currentPrice")),
                "previousClose": info.get("previousClose"),
                "open": info.get("regularMarketOpen", info.get("open")),
                "dayHigh": info.get("regularMarketDayHigh", info.get("dayHigh")),
                "dayLow": info.get("regularMarketDayLow", info.get("dayLow")),
                "volume": info.get("regularMarketVolume", info.get("volume")),
                "averageVolume": info.get("averageVolume"),
                "marketCap": info.get("marketCap"),
                "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                "currency": info.get("currency"),
                "exchange": info.get("exchange"),
            }

            # Filter out None values
            quote_data = {k: v for k, v in quote_data.items() if v is not None}

            execution_time = time.time() - start_time
            formatted_message = self._format_financial_data(quote_data, "quote")

            metadata = YFinanceMetadata(
                symbol=symbol.upper(),
                operation="get_stock_quote",
                execution_time=execution_time,
                data_points=len(quote_data),
                timestamp=datetime.now().isoformat(),
            )


            return {
                "result": formatted_message,
                "metadata": metadata.model_dump(),
                "success": True
            }

        except Exception as e:
            if self._is_timezone_error(e):
                error_msg = (
                    f"Failed to fetch stock quote for {symbol}: "
                    "Yahoo Finance did not return exchange timezone metadata for this symbol "
                    "(often caused by an invalid/delisted ticker, a Google-Finance style symbol like "
                    "'AAPL:NASDAQ', or a transient Yahoo response/rate limit)."
                )
                metadata = YFinanceMetadata(
                    symbol=(symbol or "").upper(),
                    operation="get_stock_quote",
                    error_type="missing_timezone",
                    timestamp=datetime.now().isoformat(),
                )
                return {
                    "error": error_msg,
                    "metadata": metadata.model_dump(),
                    "success": False,
                    "traceback": traceback.format_exc(),
                }
            error_msg = f"Failed to fetch stock quote for {symbol}: {str(e)}"

            metadata = YFinanceMetadata(
                symbol=symbol.upper(),
                operation="get_stock_quote",
                error_type=type(e).__name__,
                timestamp=datetime.now().isoformat(),
            )

            return {
                "error": error_msg,
                "metadata": metadata.model_dump(),
                "success": False,
                "traceback": traceback.format_exc()
            }

    def mcp_get_historical_data(
        self,
        symbol: str = Field(description="Stock ticker symbol (e.g., AAPL, MSFT)"),
        start: str = Field(description="Start date (YYYY-MM-DD)"),
        end: str = Field(description="End date (YYYY-MM-DD)"),
        interval: str = Field(default="1y", description="Data interval (supports 1d/1wk/1mo/etc; 1y returns yearly aggregated values)"),
        max_rows_preview: int = Field(default=0, description="Max rows for preview (0 for all data)"),
    ) :
        """Retrieve historical stock data.

        Fetches historical OHLCV (Open, High, Low, Close, Volume) data for the
        specified ticker symbol within the given date range and interval.

        Args:
            symbol: Stock ticker symbol
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1wk, 1mo, etc.)
            max_rows_preview: Maximum rows to show in preview

        Returns:
            ActionResponse with historical data and metadata
        """
        # Handle FieldInfo objects
        if isinstance(symbol, FieldInfo):
            symbol = symbol.default
        if isinstance(start, FieldInfo):
            start = start.default
        if isinstance(end, FieldInfo):
            end = end.default
        if isinstance(interval, FieldInfo):
            interval = interval.default
        if isinstance(max_rows_preview, FieldInfo):
            max_rows_preview = max_rows_preview.default

        try:
            start_time = time.time()
            symbol = self._normalize_symbol(symbol)
            base_interval, yearly = self._normalize_interval(interval)
            print(f"📈 Fetching historical data for: {symbol} ({start} to {end}), interval={interval}")

            with self._quiet_yfinance():
                ticker = yf.Ticker(symbol)
                hist_df = ticker.history(start=start, end=end, interval=base_interval)

            if hist_df.empty:
                # yfinance can return empty frames when Yahoo responses are blocked/empty,
                # often accompanied by internal logs like:
                # - "Expecting value: line 1 column 1 (char 0)"
                # - "No timezone found, symbol may be delisted"
                #
                # Try a direct HTTP fallback to Yahoo's chart API before failing.
                historical_data = self._fetch_chart_history_via_http(
                    symbol=symbol,
                    start=start,
                    end=end,
                    interval=base_interval,
                )
                if not historical_data:
                    raise ValueError(
                        f"No historical data found for {symbol} with start={start}, end={end}, interval={interval}"
                    )
                if yearly:
                    historical_data = self._reduce_to_yearly_max_close(historical_data)

                execution_time = time.time() - start_time
                # Format message based on data size
                if max_rows_preview == 0:
                    formatted_message = self._format_historical_records_full(historical_data)
                elif max_rows_preview > 0 and len(historical_data) > max_rows_preview:
                    preview_count = max_rows_preview // 2
                    preview_count = max(1, preview_count)
                    preview_data = historical_data[:preview_count] + historical_data[-preview_count:]
                    formatted_message = self._format_financial_data(preview_data, "historical")
                    formatted_message += (
                        f"\n\n*Note: Showing preview of {len(preview_data)} out of {len(historical_data)} total records*"
                    )
                else:
                    formatted_message = self._format_financial_data(historical_data, "historical")

                metadata = YFinanceMetadata(
                    symbol=symbol.upper(),
                    operation="get_historical_data",
                    execution_time=execution_time,
                    data_points=len(historical_data),
                    timestamp=datetime.now().isoformat(),
                )

                print(f"✅ Historical data retrieved via HTTP fallback: {len(historical_data)} records")

                return {
                    "result": formatted_message,
                    "metadata": metadata.model_dump(),
                    "success": True,
                }

            # Convert DataFrame to list of dictionaries
            hist_df.reset_index(inplace=True)

            # Ensure date columns are strings for JSON serialization
            if "Date" in hist_df.columns:
                hist_df["Date"] = hist_df["Date"].astype(str)
            if "Datetime" in hist_df.columns:
                hist_df["Datetime"] = hist_df["Datetime"].astype(str)

            # Clean column names
            hist_df.columns = hist_df.columns.str.replace(" ", "")

            historical_data = hist_df.to_dict(orient="records")
            if yearly:
                historical_data = self._reduce_to_yearly_max_close(historical_data)
            execution_time = time.time() - start_time

            # Format message based on data size
            if max_rows_preview == 0:
                formatted_message = self._format_historical_records_full(historical_data)
            elif max_rows_preview > 0 and len(historical_data) > max_rows_preview:
                preview_count = max_rows_preview // 2
                preview_count = max(1, preview_count)

                preview_data = historical_data[:preview_count] + historical_data[-preview_count:]
                formatted_message = self._format_financial_data(preview_data, "historical")
                formatted_message += (
                    f"\n\n*Note: Showing preview of {len(preview_data)} out of {len(historical_data)} total records*"
                )
            else:
                formatted_message = self._format_financial_data(historical_data, "historical")

            metadata = YFinanceMetadata(
                symbol=symbol.upper(),
                operation="get_historical_data",
                execution_time=execution_time,
                data_points=len(historical_data),
                timestamp=datetime.now().isoformat(),
            )

            print(f"✅ Historical data retrieved: {len(historical_data)} records")

            return {
                "result": formatted_message,
                "metadata": metadata.model_dump(),
                "success": True
            }

        except Exception as e:
            if self._is_timezone_error(e):
                error_msg = (
                    f"Failed to fetch historical data for {symbol}: "
                    "Yahoo Finance did not return exchange timezone metadata for this symbol "
                    "(often caused by an invalid/delisted ticker, a Google-Finance style symbol like "
                    "'AAPL:NASDAQ', or a transient Yahoo response/rate limit)."
                )
                metadata = YFinanceMetadata(
                    symbol=(symbol or "").upper(),
                    operation="get_historical_data",
                    error_type="missing_timezone",
                    timestamp=datetime.now().isoformat(),
                )
                return {
                    "error": error_msg,
                    "metadata": metadata.model_dump(),
                    "success": False,
                    "traceback": traceback.format_exc(),
                }
            error_msg = f"Failed to fetch historical data for {symbol}: {str(e)}"
            if isinstance(e, ValueError) and "Yahoo chart" in str(e):
                # Make the error more actionable: this typically indicates network block/captive portal/rate limit.
                error_msg = (
                    f"Failed to fetch historical data for {symbol}: Yahoo Finance data endpoint returned "
                    f"an unexpected response ({str(e)}). This often means outbound HTTP to "
                    "query1.finance.yahoo.com is blocked, rate-limited, or intercepted (HTML instead of JSON)."
                )

            metadata = YFinanceMetadata(
                symbol=symbol.upper(),
                operation="get_historical_data",
                error_type=type(e).__name__,
                timestamp=datetime.now().isoformat(),
            )

            return {
                "error": error_msg,
                "metadata": metadata.model_dump(),
                "success": False,
                "traceback": traceback.format_exc()
            }

    def mcp_get_company_info(
        self,
        symbol: str = Field(description="Stock ticker symbol (e.g., AAPL, MSFT)"),
    ) :
        """Get company information and business details.

        Fetches comprehensive company information including sector, industry,
        employee count, business summary, location, and other key details.

        Args:
            symbol: Stock ticker symbol

        Returns:
            ActionResponse with company information and metadata
        """
        # Handle FieldInfo objects
        if isinstance(symbol, FieldInfo):
            symbol = symbol.default

        try:
            start_time = time.time()
            symbol = self._normalize_symbol(symbol)
            print(f"🏢 Fetching company info for: {symbol}")

            with self._quiet_yfinance():
                ticker = yf.Ticker(symbol)
                info = ticker.info

            if not info or not info.get("symbol"):
                raise ValueError(f"No company information found for symbol: {symbol}. It might be invalid.")

            # Extract key company information
            company_data = {
                "symbol": info.get("symbol"),
                "shortName": info.get("shortName"),
                "longName": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "fullTimeEmployees": info.get("fullTimeEmployees"),
                "longBusinessSummary": info.get("longBusinessSummary"),
                "city": info.get("city"),
                "state": info.get("state"),
                "country": info.get("country"),
                "website": info.get("website"),
                "exchange": info.get("exchange"),
                "currency": info.get("currency"),
                "marketCap": info.get("marketCap"),
            }

            # Filter out None values
            company_data = {k: v for k, v in company_data.items() if v is not None}

            execution_time = time.time() - start_time
            formatted_message = self._format_financial_data(company_data, "company")

            metadata = YFinanceMetadata(
                symbol=symbol.upper(),
                operation="get_company_info",
                execution_time=execution_time,
                data_points=len(company_data),
                timestamp=datetime.now().isoformat(),
            )

            print("✅ Company information retrieved successfully")

            return {
                "result": formatted_message,
                "metadata": metadata.model_dump(),
                "success": True
            }

        except Exception as e:
            if self._is_timezone_error(e):
                error_msg = (
                    f"Failed to fetch company info for {symbol}: "
                    "Yahoo Finance did not return exchange timezone metadata for this symbol "
                    "(often caused by an invalid/delisted ticker, a Google-Finance style symbol like "
                    "'AAPL:NASDAQ', or a transient Yahoo response/rate limit)."
                )
                metadata = YFinanceMetadata(
                    symbol=(symbol or "").upper(),
                    operation="get_company_info",
                    error_type="missing_timezone",
                    timestamp=datetime.now().isoformat(),
                )
                return {
                    "error": error_msg,
                    "metadata": metadata.model_dump(),
                    "success": False,
                    "traceback": traceback.format_exc(),
                }
            error_msg = f"Failed to fetch company info for {symbol}: {str(e)}"

            metadata = YFinanceMetadata(
                symbol=symbol.upper(),
                operation="get_company_info",
                error_type=type(e).__name__,
                timestamp=datetime.now().isoformat(),
            )

            return {
                "error": error_msg,
                "metadata": metadata.model_dump(),
                "success": False,
                "traceback": traceback.format_exc()
            }

    def mcp_get_financial_statements(
        self,
        symbol: str = Field(description="Stock ticker symbol (e.g., AAPL, MSFT)"),
        statement_type: str = Field(description="Statement type: income_statement, balance_sheet, or cash_flow"),
        period_type: str = Field(default="annual", description="Period type: annual or quarterly"),
        max_columns_preview: int = Field(default=4, description="Max periods to show (0 for all)"),
    ) :
        """Get financial statements for a company.

        Fetches financial statements including income statement, balance sheet,
        or cash flow statement for the specified company and period.

        Args:
            symbol: Stock ticker symbol
            statement_type: Type of statement (income_statement, balance_sheet, cash_flow)
            period_type: Period type (annual or quarterly)
            max_columns_preview: Maximum periods to show in preview

        Returns:
            ActionResponse with financial statement data and metadata
        """
        # Handle FieldInfo objects
        if isinstance(symbol, FieldInfo):
            symbol = symbol.default
        if isinstance(statement_type, FieldInfo):
            statement_type = statement_type.default
        if isinstance(period_type, FieldInfo):
            period_type = period_type.default
        if isinstance(max_columns_preview, FieldInfo):
            max_columns_preview = max_columns_preview.default

        try:
            start_time = time.time()
            symbol = self._normalize_symbol(symbol)
            print(f"📋 Fetching {statement_type} for: {symbol} ({period_type})")

            with self._quiet_yfinance():
                ticker = yf.Ticker(symbol)
            statement_df = None

            # Get appropriate statement
            if statement_type == "income_statement":
                with self._quiet_yfinance():
                    statement_df = ticker.income_stmt if period_type == "annual" else ticker.quarterly_income_stmt
            elif statement_type == "balance_sheet":
                with self._quiet_yfinance():
                    statement_df = ticker.balance_sheet if period_type == "annual" else ticker.quarterly_balance_sheet
            elif statement_type == "cash_flow":
                with self._quiet_yfinance():
                    statement_df = ticker.cashflow if period_type == "annual" else ticker.quarterly_cashflow
            else:
                raise ValueError(
                    f"Invalid statement_type: {statement_type}. "
                    "Must be one of: income_statement, balance_sheet, cash_flow"
                )

            if statement_df is None or statement_df.empty:
                raise ValueError(f"No {period_type} {statement_type} data found for symbol {symbol}")

            # Process DataFrame
            statement_df.reset_index(inplace=True)
            statement_df.rename(columns={"index": "Item"}, inplace=True)

            # Convert date columns to strings
            for col in statement_df.columns:
                if col != "Item":
                    try:
                        if hasattr(col, "strftime"):
                            statement_df.rename(columns={col: col.strftime("%Y-%m-%d")}, inplace=True)
                    except Exception:
                        pass

            statement_data = statement_df.to_dict(orient="records")
            execution_time = time.time() - start_time

            # Format message
            if max_columns_preview > 0 and len(statement_df.columns) > (max_columns_preview + 1):
                columns_to_keep = ["Item"] + list(statement_df.columns[1 : max_columns_preview + 1])
                preview_df = statement_df[columns_to_keep]
                preview_data = preview_df.to_dict(orient="records")

                formatted_message = f"# {statement_type.replace('_', ' ').title()} ({period_type.title()})\n\n"
                formatted_message += (
                    f"Showing preview of most recent {max_columns_preview} periods "
                    f"out of {len(statement_df.columns) - 1} available.\n\n"
                )

                # Show key financial items
                for item in preview_data[:10]:  # Show first 10 items
                    item_name = item.get("Item", "N/A")
                    formatted_message += f"**{item_name}:**\n"
                    for col, value in item.items():
                        if col != "Item" and value is not None:
                            formatted_message += (
                                f"  - {col}: {value:,}\n"
                                if isinstance(value, (int, float))
                                else f"  - {col}: {value}\n"
                            )
                    formatted_message += "\n"
            else:
                formatted_message = f"# {statement_type.replace('_', ' ').title()} ({period_type.title()})\n\n"
                formatted_message += f"Complete financial statement with {len(statement_data)} line items.\n"

            metadata = YFinanceMetadata(
                symbol=symbol.upper(),
                operation="get_financial_statements",
                execution_time=execution_time,
                data_points=len(statement_data),
                timestamp=datetime.now().isoformat(),
            )

            print(f"✅ Financial statements retrieved: {len(statement_data)} items")

            return {
                "result": formatted_message,
                "metadata": metadata.model_dump(),
                "success": True
            }

        except Exception as e:
            if self._is_timezone_error(e):
                error_msg = (
                    f"Failed to fetch {statement_type} for {symbol}: "
                    "Yahoo Finance did not return exchange timezone metadata for this symbol "
                    "(often caused by an invalid/delisted ticker, a Google-Finance style symbol like "
                    "'AAPL:NASDAQ', or a transient Yahoo response/rate limit)."
                )
                metadata = YFinanceMetadata(
                    symbol=(symbol or "").upper(),
                    operation="get_financial_statements",
                    error_type="missing_timezone",
                    timestamp=datetime.now().isoformat(),
                )
                return {
                    "error": error_msg,
                    "metadata": metadata.model_dump(),
                    "success": False,
                    "traceback": traceback.format_exc(),
                }
            error_msg = f"Failed to fetch {statement_type} for {symbol}: {str(e)}"

            metadata = YFinanceMetadata(
                symbol=symbol.upper(),
                operation="get_financial_statements",
                error_type=type(e).__name__,
                timestamp=datetime.now().isoformat(),
            )

            return {
                "error": error_msg,
                "metadata": metadata.model_dump(),
                "success": False,
                "traceback": traceback.format_exc()
            }


# Default arguments for testing
if __name__ == "__main__":
    """Test all Yahoo Finance functions."""
    tool = Yahoo_Finance_Tool()
    # tool.embed_tool()
    print(tool.run(operation="get_historical_data", symbol="AAPL", start_date="1980-01-01", end_date="2024-12-31"))