#!/usr/bin/env python3
"""Research Assistant Agent - OpenAI Format Tool Ingestion Demo.

This agent helps users research topics by:
1. Searching multiple sources (web, academic papers, news)
2. Extracting and summarizing key information
3. Organizing findings into structured notes
4. Generating a final research report

This demonstrates ingesting OpenAI function-calling format tools into
amla-sandbox, showing how existing OpenAI tool definitions can be
reused with Amla's secure sandbox execution.

Real-world use case: Automating literature reviews, competitive analysis,
or any research task that requires gathering and synthesizing information
from multiple sources.

Requirements:
    uv pip install "git+https://github.com/amlalabs/amla-sandbox#subdirectory=src/python/packages/amla-sandbox"
    uv pip install openai python-dotenv

Usage:
    export AMLA_WASM_PATH=/path/to/amla_sandbox.wasm
    python research_assistant.py
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from amla_sandbox import create_sandbox_tool
from amla_sandbox.tools import from_openai_tools

# =============================================================================
# Configuration
# =============================================================================

# Load environment variables from .env file
# Searches up the directory tree to find .env
env_path = Path(__file__).parent
while env_path != env_path.parent:
    if (env_path / ".env").exists():
        load_dotenv(env_path / ".env")
        break
    env_path = env_path.parent

# Set WASM path if not already set
if "AMLA_WASM_PATH" not in os.environ:
    wasm_path = Path(__file__).parents[5] / "rust/amla-sandbox/pkg/amla_sandbox_bg.wasm"
    if wasm_path.exists():
        os.environ["AMLA_WASM_PATH"] = str(wasm_path)


# =============================================================================
# Tool Definitions (OpenAI Function Calling Format)
# =============================================================================

# These are defined in the exact format that OpenAI's API expects.
# In a real application, these might come from an existing OpenAI integration.

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information on a topic. Returns relevant snippets and URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_academic",
            "description": "Search academic papers and journals. Returns paper titles, authors, and abstracts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Academic search query",
                    },
                    "year_from": {
                        "type": "integer",
                        "description": "Filter papers from this year onwards",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and extract text content from a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return",
                        "default": 5000,
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a research note with tags for organization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Note title",
                    },
                    "content": {
                        "type": "string",
                        "description": "Note content (markdown supported)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization",
                    },
                    "source_url": {
                        "type": "string",
                        "description": "Source URL if applicable",
                    },
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_notes",
            "description": "Retrieve saved research notes, optionally filtered by tag.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "Filter by tag (optional)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "Generate a structured research report from saved notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Report title",
                    },
                    "sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Section headings to include",
                    },
                },
                "required": ["title"],
            },
        },
    },
]


# =============================================================================
# Tool Implementations
# =============================================================================

# In-memory storage for research notes (in production, use a database)
_notes_storage: list[dict[str, Any]] = []


def search_web(query: str, num_results: int = 5) -> dict[str, Any]:
    """Search the web for information.

    In a real implementation, this would use a search API like:
    - Google Custom Search API
    - Bing Search API
    - SerpAPI
    - Tavily (designed for AI agents)
    """
    # Mock search results
    # In production, replace with actual API calls
    results = [
        {
            "title": f"Understanding {query}: A Comprehensive Guide",
            "url": f"https://example.com/guide/{query.replace(' ', '-').lower()}",
            "snippet": f"This comprehensive guide covers everything you need to know about {query}. "
            f"From basics to advanced concepts, learn how {query} impacts modern technology.",
        },
        {
            "title": f"Latest Research on {query} (2024)",
            "url": f"https://research.example.com/{query.replace(' ', '-').lower()}",
            "snippet": f"Recent studies show significant advances in {query}. "
            f"Researchers have discovered new applications and improvements.",
        },
        {
            "title": f"{query} Best Practices and Implementation",
            "url": f"https://dev.example.com/best-practices/{query.replace(' ', '-').lower()}",
            "snippet": f"Learn industry best practices for implementing {query}. "
            f"Includes code examples, case studies, and common pitfalls to avoid.",
        },
    ][:num_results]

    return {
        "query": query,
        "num_results": len(results),
        "results": results,
    }


def search_academic(query: str, year_from: int | None = None) -> dict[str, Any]:
    """Search academic papers.

    In production, integrate with:
    - Semantic Scholar API
    - arXiv API
    - Google Scholar (via SerpAPI)
    - PubMed API
    """
    year_filter = f" (from {year_from})" if year_from else ""

    # Mock academic results
    papers = [
        {
            "title": f"A Survey of {query} Methods and Applications",
            "authors": ["Smith, J.", "Johnson, A.", "Williams, R."],
            "year": 2024,
            "abstract": f"This paper presents a comprehensive survey of {query}. "
            f"We analyze 150+ papers and identify key trends and future directions.",
            "citations": 42,
            "url": "https://arxiv.org/abs/2024.12345",
        },
        {
            "title": f"Novel Approaches to {query}: A Benchmark Study",
            "authors": ["Chen, L.", "Kumar, S."],
            "year": 2023,
            "abstract": f"We propose three novel approaches to {query} and evaluate them "
            f"on standard benchmarks, achieving state-of-the-art results.",
            "citations": 89,
            "url": "https://arxiv.org/abs/2023.67890",
        },
    ]

    if year_from:
        papers = [p for p in papers if p["year"] >= year_from]

    return {
        "query": query + year_filter,
        "num_papers": len(papers),
        "papers": papers,
    }


def fetch_url(url: str, max_chars: int = 5000) -> dict[str, Any]:
    """Fetch and extract text from a URL.

    In production, use:
    - requests + BeautifulSoup for HTML
    - trafilatura for article extraction
    - Jina Reader API for clean extraction
    """
    # Mock URL fetch
    # In production, actually fetch and parse the URL

    # Extract domain for realistic simulation
    domain = url.split("/")[2] if "/" in url else "example.com"

    content = f"""
    # Content from {domain}

    This is the extracted content from the URL: {url}

    ## Key Points

    1. The article discusses important concepts related to the topic.
    2. Several examples and case studies are provided.
    3. The author presents both advantages and challenges.
    4. Future directions and recommendations are outlined.

    ## Summary

    The content provides valuable insights that can be used for research purposes.
    Main takeaways include practical applications and theoretical foundations.
    """

    return {
        "url": url,
        "title": f"Article from {domain}",
        "content": content[:max_chars],
        "chars_extracted": min(len(content), max_chars),
    }


def save_note(
    title: str,
    content: str,
    tags: list[str] | None = None,
    source_url: str | None = None,
) -> dict[str, Any]:
    """Save a research note."""
    note = {
        "id": len(_notes_storage) + 1,
        "title": title,
        "content": content,
        "tags": tags or [],
        "source_url": source_url,
        "created_at": datetime.now().isoformat(),
    }
    _notes_storage.append(note)

    return {
        "status": "saved",
        "note_id": note["id"],
        "title": title,
        "tags": tags or [],
    }


def get_notes(tag: str | None = None) -> dict[str, Any]:
    """Retrieve saved notes, optionally filtered by tag."""
    if tag:
        filtered = [n for n in _notes_storage if tag in n.get("tags", [])]
    else:
        filtered = _notes_storage

    return {
        "filter": tag,
        "count": len(filtered),
        "notes": [
            {
                "id": n["id"],
                "title": n["title"],
                "tags": n["tags"],
                "preview": n["content"][:200] + "..."
                if len(n["content"]) > 200
                else n["content"],
            }
            for n in filtered
        ],
    }


def generate_report(title: str, sections: list[str] | None = None) -> dict[str, Any]:
    """Generate a research report from saved notes."""
    if not _notes_storage:
        return {"error": "No notes saved. Please save some research notes first."}

    # Default sections if none provided
    if not sections:
        sections = ["Introduction", "Key Findings", "Analysis", "Conclusion"]

    # Build report from notes
    report_parts = [
        f"# {title}\n",
        f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
    ]

    for section in sections:
        report_parts.append(f"\n## {section}\n")

        # Add relevant notes to each section
        for note in _notes_storage:
            report_parts.append(f"\n### {note['title']}\n")
            report_parts.append(note["content"])
            if note.get("source_url"):
                report_parts.append(f"\n*Source: {note['source_url']}*\n")

    report_content = "\n".join(report_parts)

    return {
        "title": title,
        "sections": sections,
        "word_count": len(report_content.split()),
        "notes_included": len(_notes_storage),
        "report": report_content,
    }


# =============================================================================
# Agent Setup
# =============================================================================


def create_research_agent() -> tuple[Any, Any]:
    """Create the research assistant agent.

    Returns:
        Tuple of (bash_tool, openai_client) for running the agent.
    """
    # Map tool names to their implementations
    handlers = {
        "search_web": search_web,
        "search_academic": search_academic,
        "fetch_url": fetch_url,
        "save_note": save_note,
        "get_notes": get_notes,
        "generate_report": generate_report,
    }

    # Ingest OpenAI format tools into amla-sandbox format
    funcs, defs = from_openai_tools(OPENAI_TOOLS, handlers=handlers)

    print(f"✓ Ingested {len(defs)} tools from OpenAI format:")
    for d in defs:
        print(f"  • {d.name}: {d.description[:60]}...")

    # Create the sandbox tool with safety constraints
    sandbox = create_sandbox_tool(
        tools=funcs,
        # Limit tool calls to prevent runaway agents
        max_calls={
            "search_web": 10,
            "search_academic": 10,
            "fetch_url": 20,
            "save_note": 50,
            "get_notes": 20,
            "generate_report": 5,
        },
    )

    # Initialize OpenAI client
    client = OpenAI()

    return sandbox, client


def run_agent(sandbox: Any, client: Any, user_request: str) -> str:
    """Run the research agent with a user request.

    The agent uses a code-generation approach:
    1. LLM generates JavaScript code to accomplish the task
    2. Code executes in the sandbox with access to tools
    3. Results are processed and returned

    This is more efficient than traditional tool-calling loops because
    the agent can make multiple tool calls in a single execution.
    """

    # System prompt that teaches the agent how to use the sandbox
    system_prompt = """Write CONCISE JavaScript (15 lines max). Tools (use await):
- search_web({query}) - returns {results: [{title, snippet, url}]}
- save_note({title, content, tags}) - saves a note
- generate_report({title}) - returns {report: string}

Use console.log() for output. No markdown explanation."""

    # Get code from the LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Cost-effective for code generation
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_request},
        ],
        temperature=0.7,
    )

    # Extract the code from the response
    assistant_message = response.choices[0].message.content

    # Parse out the JavaScript code block
    code_match = re.search(
        r"```(?:javascript|js)?\n(.*?)```", assistant_message, re.DOTALL
    )
    if code_match:
        code = code_match.group(1)
    else:
        # If no code block, treat the whole response as code
        code = assistant_message

    print("\n📝 Agent generated code:")
    print("-" * 40)
    print(code[:500] + "..." if len(code) > 500 else code)
    print("-" * 40)

    # Execute in the sandbox
    print("\n⚡ Executing in sandbox...")
    result = sandbox.run(code, language="javascript")

    return result


# =============================================================================
# Main
# =============================================================================


def main():
    """Run the research assistant demo."""
    print("=" * 60)
    print("Research Assistant Agent")
    print("OpenAI Format Tool Ingestion Demo")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY not set")
        print("Please set it in your environment or .env file")
        return

    # Create the agent
    print("\n🔧 Setting up agent...")
    sandbox, client = create_research_agent()

    # Example research request
    research_topic = "WebAssembly for AI agent sandboxing"

    print(f"\n🔍 Research request: '{research_topic}'")
    print("\n" + "=" * 60)

    # Run the agent
    result = run_agent(
        sandbox,
        client,
        f"Search for '{research_topic}', save one note, generate a short report.",
    )

    print("\n📊 Results:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
