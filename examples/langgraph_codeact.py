#!/usr/bin/env python3
"""LangGraph CodeAct: Complex Multi-Step Agents with Code Generation

This example demonstrates the CodeAct pattern where the LLM writes JavaScript
code to orchestrate multiple tool calls, process data, and make decisions.

This is where amla-sandbox shines - the LLM can:
- Chain multiple tool calls in a single turn
- Use loops and conditionals
- Process intermediate data with JavaScript
- Pipe results through shell commands
- Store/retrieve data from the virtual filesystem

Prerequisites:
    pip install "amla-sandbox[langgraph]"
    export OPENAI_API_KEY=your-key-here

Run:
    python langgraph_codeact.py              # Run all examples
    python langgraph_codeact.py --example 1  # Run specific example
    python langgraph_codeact.py --list       # List examples
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Rich Tool Set for Complex Demos
# =============================================================================


def get_stock_price(symbol: str) -> dict[str, Any]:
    """Get current stock price and daily change.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "GOOGL", "MSFT", "NVDA")
    """
    stocks = {
        "AAPL": {"price": 178.50, "change": 2.30, "volume": 45_000_000},
        "GOOGL": {"price": 141.25, "change": -0.75, "volume": 22_000_000},
        "MSFT": {"price": 378.90, "change": 4.20, "volume": 18_000_000},
        "NVDA": {"price": 721.33, "change": 15.40, "volume": 35_000_000},
        "AMZN": {"price": 178.25, "change": -1.20, "volume": 28_000_000},
        "META": {"price": 505.75, "change": 8.90, "volume": 15_000_000},
        "TSLA": {"price": 248.50, "change": -5.30, "volume": 95_000_000},
    }
    symbol = symbol.upper()
    if symbol in stocks:
        return {
            "symbol": symbol,
            **stocks[symbol],
            "timestamp": datetime.now().isoformat(),
        }
    return {"symbol": symbol, "error": "Symbol not found"}


def get_company_info(symbol: str) -> dict[str, Any]:
    """Get company information and sector.

    Args:
        symbol: Stock ticker symbol
    """
    companies = {
        "AAPL": {"name": "Apple Inc.", "sector": "Technology", "employees": 164000},
        "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "employees": 182000},
        "MSFT": {
            "name": "Microsoft Corp.",
            "sector": "Technology",
            "employees": 221000,
        },
        "NVDA": {"name": "NVIDIA Corp.", "sector": "Technology", "employees": 29600},
        "AMZN": {
            "name": "Amazon.com Inc.",
            "sector": "Consumer Cyclical",
            "employees": 1540000,
        },
        "META": {
            "name": "Meta Platforms Inc.",
            "sector": "Technology",
            "employees": 67317,
        },
        "TSLA": {"name": "Tesla Inc.", "sector": "Automotive", "employees": 140000},
    }
    symbol = symbol.upper()
    if symbol in companies:
        return {"symbol": symbol, **companies[symbol]}
    return {"symbol": symbol, "error": "Company not found"}


def get_analyst_rating(symbol: str) -> dict[str, Any]:
    """Get analyst consensus rating and price target.

    Args:
        symbol: Stock ticker symbol
    """
    ratings = {
        "AAPL": {"rating": "Buy", "target": 200.00, "analysts": 45},
        "GOOGL": {"rating": "Strong Buy", "target": 165.00, "analysts": 52},
        "MSFT": {"rating": "Strong Buy", "target": 420.00, "analysts": 48},
        "NVDA": {"rating": "Strong Buy", "target": 850.00, "analysts": 55},
        "AMZN": {"rating": "Buy", "target": 210.00, "analysts": 50},
        "META": {"rating": "Buy", "target": 550.00, "analysts": 42},
        "TSLA": {"rating": "Hold", "target": 265.00, "analysts": 38},
    }
    symbol = symbol.upper()
    if symbol in ratings:
        return {"symbol": symbol, **ratings[symbol]}
    return {"symbol": symbol, "error": "No rating available"}


def search_news(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search financial news articles.

    Args:
        query: Search query (company name, ticker, or topic)
        limit: Maximum articles to return
    """
    all_news = [
        {
            "title": "NVIDIA Announces New AI Chip",
            "source": "Reuters",
            "sentiment": "positive",
            "ticker": "NVDA",
        },
        {
            "title": "Apple Vision Pro Sales Exceed Expectations",
            "source": "Bloomberg",
            "sentiment": "positive",
            "ticker": "AAPL",
        },
        {
            "title": "Microsoft Azure Revenue Grows 29%",
            "source": "CNBC",
            "sentiment": "positive",
            "ticker": "MSFT",
        },
        {
            "title": "Tesla Recalls 2M Vehicles",
            "source": "WSJ",
            "sentiment": "negative",
            "ticker": "TSLA",
        },
        {
            "title": "Meta's AI Assistant Reaches 100M Users",
            "source": "TechCrunch",
            "sentiment": "positive",
            "ticker": "META",
        },
        {
            "title": "Amazon Expands Same-Day Delivery",
            "source": "Reuters",
            "sentiment": "positive",
            "ticker": "AMZN",
        },
        {
            "title": "Google Antitrust Trial Begins",
            "source": "NYT",
            "sentiment": "negative",
            "ticker": "GOOGL",
        },
        {
            "title": "Tech Sector Outlook Remains Strong",
            "source": "Barrons",
            "sentiment": "positive",
            "ticker": None,
        },
    ]
    query_lower = query.lower()
    matches = [
        n
        for n in all_news
        if query_lower in str(n["title"]).lower()
        or (n["ticker"] is not None and query_lower in str(n["ticker"]).lower())
    ]
    return matches[:limit] if matches else all_news[:limit]


def execute_trade(symbol: str, action: str, quantity: int) -> dict[str, Any]:
    """Execute a simulated trade (buy/sell).

    Args:
        symbol: Stock ticker symbol
        action: Either "buy" or "sell"
        quantity: Number of shares
    """
    if action.lower() not in ("buy", "sell"):
        return {"error": "Action must be 'buy' or 'sell'"}

    price_data = get_stock_price(symbol)
    if "error" in price_data:
        return price_data

    total = price_data["price"] * quantity
    return {
        "order_id": f"ORD-{abs(hash(symbol + action)) % 100000:05d}",
        "symbol": symbol.upper(),
        "action": action.lower(),
        "quantity": quantity,
        "price": price_data["price"],
        "total": round(total, 2),
        "status": "executed",
        "timestamp": datetime.now().isoformat(),
    }


def get_portfolio() -> dict[str, Any]:
    """Get current portfolio holdings and value."""
    return {
        "holdings": [
            {"symbol": "AAPL", "shares": 100, "avg_cost": 165.00},
            {"symbol": "MSFT", "shares": 50, "avg_cost": 350.00},
            {"symbol": "NVDA", "shares": 25, "avg_cost": 450.00},
        ],
        "cash": 25000.00,
        "last_updated": datetime.now().isoformat(),
    }


def calculate_metrics(prices: list[float]) -> dict[str, float | str]:
    """Calculate statistical metrics for a list of prices.

    Args:
        prices: List of price values
    """
    if not prices:
        return {"error": "No prices provided"}

    n = len(prices)
    mean = sum(prices) / n
    variance = sum((p - mean) ** 2 for p in prices) / n
    std_dev = variance**0.5

    return {
        "count": n,
        "mean": round(mean, 2),
        "min": round(min(prices), 2),
        "max": round(max(prices), 2),
        "std_dev": round(std_dev, 2),
        "range": round(max(prices) - min(prices), 2),
    }


# =============================================================================
# Examples
# =============================================================================


def example_1_portfolio_analysis() -> None:
    """Example 1: Multi-step portfolio analysis with code generation."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from amla_sandbox import create_sandbox_tool

    print("\n" + "=" * 60)
    print("Example 1: Portfolio Analysis (Code Generation)")
    print("=" * 60)
    print("""
This example shows the LLM writing JavaScript to:
- Fetch portfolio holdings
- Loop through each holding to get current prices
- Calculate gains/losses
- Generate a summary report
""")

    sandbox = create_sandbox_tool(
        tools=[get_portfolio, get_stock_price, get_company_info, calculate_metrics],
        max_calls=50,
    )

    model = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-5"),
        temperature=0,
    )

    # Use code generation approach
    agent: Any = create_react_agent(
        model, [sandbox.as_langchain_tool()], prompt=sandbox.get_system_prompt()
    )

    print("-" * 40)
    print(
        "Query: Analyze my portfolio - show current value, gains/losses, and sector breakdown"
    )
    print("-" * 40)

    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "Analyze my portfolio. For each holding, get the current price and company info. "
                    "Calculate the total value, gain/loss for each position, and group by sector. "
                    "Show me a summary with the total portfolio value and overall gain/loss percentage.",
                )
            ]
        },
        config={"recursion_limit": 25},
    )

    # Show the code that was generated
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "sandbox":
                    print("\n[Generated Code]:")
                    code = tc.get("args", {}).get("code", "")
                    print(code[:500] + "..." if len(code) > 500 else code)
        if msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", ""):
            print(f"\nAssistant: {msg.content}")

    print("\n" + "-" * 40)
    print("Example 1 completed!")


def example_2_stock_screener() -> None:
    """Example 2: Stock screening with filtering and sorting."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from amla_sandbox import create_sandbox_tool

    print("\n" + "=" * 60)
    print("Example 2: Stock Screener (Loop + Filter + Sort)")
    print("=" * 60)
    print("""
This example shows the LLM writing JavaScript to:
- Fetch data for multiple stocks in a loop
- Filter based on criteria
- Sort results
- Use the VFS to store intermediate data
""")

    sandbox = create_sandbox_tool(
        tools=[get_stock_price, get_analyst_rating, get_company_info],
        max_calls=50,
    )

    model = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-5"),
        temperature=0,
    )

    agent: Any = create_react_agent(
        model, [sandbox.as_langchain_tool()], prompt=sandbox.get_system_prompt()
    )

    print("-" * 40)
    print("Query: Screen tech stocks for best opportunities")
    print("-" * 40)

    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "Screen these stocks: AAPL, GOOGL, MSFT, NVDA, META, AMZN, TSLA. "
                    "For each one, get the price, analyst rating, and company info. "
                    "Filter to only show stocks with 'Buy' or 'Strong Buy' ratings. "
                    "Sort by the upside potential (target price vs current price). "
                    "Show me the top opportunities with their upside percentage.",
                )
            ]
        },
        config={"recursion_limit": 25},
    )

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "sandbox":
                    print("\n[Generated Code]:")
                    code = tc.get("args", {}).get("code", "")
                    print(code[:600] + "..." if len(code) > 600 else code)
        if msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", ""):
            print(f"\nAssistant: {msg.content}")

    print("\n" + "-" * 40)
    print("Example 2 completed!")


def example_3_news_sentiment_analysis() -> None:
    """Example 3: News aggregation with sentiment analysis."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from amla_sandbox import create_sandbox_tool

    print("\n" + "=" * 60)
    print("Example 3: News Sentiment Analysis (Aggregation + Shell)")
    print("=" * 60)
    print("""
This example shows the LLM:
- Fetching news for multiple stocks
- Aggregating sentiment scores
- Using shell commands (jq) for data processing
- Writing results to VFS and reading back
""")

    sandbox = create_sandbox_tool(
        tools=[search_news, get_stock_price],
        max_calls=50,
    )

    model = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-5"),
        temperature=0,
    )

    agent: Any = create_react_agent(
        model, [sandbox.as_langchain_tool()], prompt=sandbox.get_system_prompt()
    )

    print("-" * 40)
    print("Query: Analyze news sentiment for my watchlist")
    print("-" * 40)

    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "I want to understand the news sentiment for NVDA, TSLA, and GOOGL. "
                    "Search news for each, count positive vs negative articles, "
                    "and correlate with today's price movement. "
                    "Save the raw data to /workspace/sentiment.json and give me a summary. "
                    "Which stock has the best sentiment-to-price alignment?",
                )
            ]
        },
        config={"recursion_limit": 25},
    )

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "sandbox":
                    print("\n[Generated Code]:")
                    code = tc.get("args", {}).get("code", "")
                    print(code[:600] + "..." if len(code) > 600 else code)
        if msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", ""):
            print(f"\nAssistant: {msg.content}")

    print("\n" + "-" * 40)
    print("Example 3 completed!")


def example_4_trading_strategy() -> None:
    """Example 4: Conditional trading based on analysis."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from amla_sandbox import create_sandbox_tool

    print("\n" + "=" * 60)
    print("Example 4: Trading Strategy (Conditionals + Execution)")
    print("=" * 60)
    print("""
This example shows the LLM writing code that:
- Analyzes current holdings
- Checks analyst ratings
- Makes conditional decisions
- Executes trades based on logic
""")

    sandbox = create_sandbox_tool(
        tools=[get_portfolio, get_stock_price, get_analyst_rating, execute_trade],
        max_calls=30,
        constraints={
            "execute_trade": {
                "quantity": "<=100",  # Max 100 shares per trade
            }
        },
    )

    model = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-5"),
        temperature=0,
    )

    agent: Any = create_react_agent(
        model, [sandbox.as_langchain_tool()], prompt=sandbox.get_system_prompt()
    )

    print("-" * 40)
    print("Query: Rebalance portfolio based on analyst ratings")
    print("-" * 40)

    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "Review my portfolio. For any holding where the analyst rating is 'Hold' or worse, "
                    "sell 10 shares. For any 'Strong Buy' rated stock, if I have cash, buy 5 more shares. "
                    "Show me what trades you would make and execute them.",
                )
            ]
        },
        config={"recursion_limit": 30},
    )

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "sandbox":
                    print("\n[Generated Code]:")
                    code = tc.get("args", {}).get("code", "")
                    print(code[:700] + "..." if len(code) > 700 else code)
        if msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", ""):
            print(f"\nAssistant: {msg.content}")

    print("\n" + "-" * 40)
    print("Example 4 completed!")


def example_5_data_pipeline() -> None:
    """Example 5: Complex data pipeline with shell integration."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from amla_sandbox import create_sandbox_tool

    print("\n" + "=" * 60)
    print("Example 5: Data Pipeline (JS + Shell + VFS)")
    print("=" * 60)
    print("""
This example shows a full data pipeline:
- Collect data with JavaScript
- Store in VFS as JSON
- Process with shell (jq, sort, etc.)
- Generate formatted output
""")

    sandbox = create_sandbox_tool(
        tools=[
            get_stock_price,
            get_company_info,
            get_analyst_rating,
            calculate_metrics,
        ],
        max_calls=50,
    )

    model = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-5"),
        temperature=0,
    )

    agent: Any = create_react_agent(
        model, [sandbox.as_langchain_tool()], prompt=sandbox.get_system_prompt()
    )

    print("-" * 40)
    print("Query: Build a stock comparison report")
    print("-" * 40)

    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    "Build me a comparison report for AAPL, MSFT, and GOOGL. "
                    "First, collect all the data (price, company info, ratings) and save to /workspace/stocks.json. "
                    "Then use shell commands to: "
                    "1) Extract just the prices and calculate the average using jq "
                    "2) Sort the stocks by price "
                    "3) Create a simple text report showing each stock's key metrics",
                )
            ]
        },
        config={"recursion_limit": 30},
    )

    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "sandbox":
                    lang = tc.get("args", {}).get("language", "javascript")
                    code = tc.get("args", {}).get("code", "")
                    print(f"\n[Generated {lang.upper()}]:")
                    print(code[:500] + "..." if len(code) > 500 else code)
        if msg.__class__.__name__ == "AIMessage" and getattr(msg, "content", ""):
            print(f"\nAssistant: {msg.content}")

    print("\n" + "-" * 40)
    print("Example 5 completed!")


# =============================================================================
# Main
# =============================================================================


EXAMPLES = {
    1: ("Portfolio Analysis", example_1_portfolio_analysis),
    2: ("Stock Screener", example_2_stock_screener),
    3: ("News Sentiment", example_3_news_sentiment_analysis),
    4: ("Trading Strategy", example_4_trading_strategy),
    5: ("Data Pipeline", example_5_data_pipeline),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LangGraph CodeAct examples - complex multi-step agents"
    )
    parser.add_argument("--example", "-e", type=int, help="Run specific example (1-5)")
    parser.add_argument("--list", "-l", action="store_true", help="List examples")
    args = parser.parse_args()

    if args.list:
        print("Available examples:")
        for num, (name, _) in EXAMPLES.items():
            print(f"  {num}. {name}")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        print("\nSet it with: export OPENAI_API_KEY=your-key-here")
        sys.exit(1)

    print("=" * 60)
    print("LangGraph CodeAct: Complex Multi-Step Agents")
    print("=" * 60)
    print(f"\nUsing model: {os.environ.get('OPENAI_MODEL', 'gpt-5')}")
    print("""
These examples demonstrate the CodeAct pattern where the LLM writes
JavaScript code to orchestrate multiple operations in a single turn.

This is more token-efficient than structured tools when you need to:
- Loop through items
- Make conditional decisions
- Chain multiple tool calls
- Process intermediate data
""")

    try:
        if args.example:
            if args.example not in EXAMPLES:
                print(f"Error: Unknown example {args.example}")
                sys.exit(1)
            name, func = EXAMPLES[args.example]
            print(f"\nRunning: {name}")
            func()
        else:
            for num, (name, func) in EXAMPLES.items():
                print(f"\n{'=' * 60}")
                print(f"Running example {num}: {name}")
                func()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
