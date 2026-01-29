#!/usr/bin/env python3
"""Data Analyst Agent - Anthropic Format Tool Ingestion Demo.

This agent helps users analyze data by:
1. Querying databases and data sources
2. Performing statistical analysis
3. Creating visualizations (as text-based charts)
4. Generating insights and recommendations

This demonstrates ingesting Anthropic tool format into amla-sandbox,
showing how existing Claude/Anthropic tool definitions can be reused
with Amla's secure sandbox execution.

Real-world use case: Business intelligence, data exploration, automated
reporting, and ad-hoc analysis without writing SQL or Python.

Requirements:
    pip install amla-sandbox openai python-dotenv

Usage:
    export AMLA_WASM_PATH=/path/to/amla_sandbox.wasm
    python data_analyst.py
"""

from __future__ import annotations

import os
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from amla_sandbox import create_sandbox_tool
from amla_sandbox.tools import from_anthropic_tools

# =============================================================================
# Configuration
# =============================================================================

# Load environment variables
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
# Mock Database
# =============================================================================

# Sample e-commerce data for demonstration
# In production, these would query actual databases


def _generate_sample_data() -> dict[str, list[dict[str, Any]]]:
    """Generate realistic sample e-commerce data."""
    random.seed(42)  # For reproducibility

    # Products
    products = [
        {"id": 1, "name": "Laptop Pro 15", "category": "Electronics", "price": 1299.99},
        {"id": 2, "name": "Wireless Mouse", "category": "Electronics", "price": 29.99},
        {"id": 3, "name": "USB-C Hub", "category": "Electronics", "price": 49.99},
        {"id": 4, "name": "Standing Desk", "category": "Furniture", "price": 599.99},
        {"id": 5, "name": "Ergonomic Chair", "category": "Furniture", "price": 449.99},
        {"id": 6, "name": "Monitor 27inch", "category": "Electronics", "price": 349.99},
        {
            "id": 7,
            "name": "Keyboard Mechanical",
            "category": "Electronics",
            "price": 149.99,
        },
        {"id": 8, "name": "Webcam HD", "category": "Electronics", "price": 79.99},
        {"id": 9, "name": "Desk Lamp", "category": "Furniture", "price": 39.99},
        {
            "id": 10,
            "name": "Cable Organizer",
            "category": "Accessories",
            "price": 14.99,
        },
    ]

    # Generate orders over the past 90 days
    orders = []
    base_date = datetime.now() - timedelta(days=90)
    regions = ["North", "South", "East", "West"]

    for i in range(500):
        order_date = base_date + timedelta(days=random.randint(0, 90))
        product = random.choice(products)
        quantity = random.randint(1, 5)

        orders.append(
            {
                "id": i + 1,
                "date": order_date.strftime("%Y-%m-%d"),
                "product_id": product["id"],
                "product_name": product["name"],
                "category": product["category"],
                "quantity": quantity,
                "unit_price": product["price"],
                "total": round(product["price"] * quantity, 2),
                "region": random.choice(regions),
                "customer_id": random.randint(1, 100),
            }
        )

    # Customer data
    customers = [
        {
            "id": i,
            "name": f"Customer {i}",
            "segment": random.choice(["Enterprise", "SMB", "Consumer"]),
        }
        for i in range(1, 101)
    ]

    return {
        "products": products,
        "orders": orders,
        "customers": customers,
    }


# Initialize sample data
SAMPLE_DATA = _generate_sample_data()


# =============================================================================
# Tool Definitions (Anthropic Format)
# =============================================================================

# These are defined in the exact format that Anthropic's API expects.
# Note: Anthropic uses `input_schema` instead of `parameters`

ANTHROPIC_TOOLS = [
    {
        "name": "query_database",
        "description": "Execute a SQL-like query against the database. Supports SELECT, WHERE, GROUP BY, ORDER BY, and LIMIT clauses. Tables: orders, products, customers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Table to query: 'orders', 'products', or 'customers'",
                    "enum": ["orders", "products", "customers"],
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to select. Use '*' for all columns.",
                },
                "where": {
                    "type": "object",
                    "description": "Filter conditions as key-value pairs. Supports: =, >, <, >=, <=, contains",
                },
                "group_by": {
                    "type": "string",
                    "description": "Column to group results by",
                },
                "order_by": {
                    "type": "string",
                    "description": "Column to sort by (prefix with '-' for descending)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of rows to return",
                },
            },
            "required": ["table"],
        },
    },
    {
        "name": "calculate_statistics",
        "description": "Calculate statistical measures for a numeric column in the data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Table to analyze",
                    "enum": ["orders", "products", "customers"],
                },
                "column": {
                    "type": "string",
                    "description": "Numeric column to analyze (e.g., 'total', 'quantity', 'price')",
                },
                "group_by": {
                    "type": "string",
                    "description": "Optional column to group statistics by",
                },
            },
            "required": ["table", "column"],
        },
    },
    {
        "name": "create_chart",
        "description": "Create a text-based visualization of data. Useful for showing trends and comparisons.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "description": "Type of chart to create",
                    "enum": ["bar", "line", "histogram"],
                },
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "value": {"type": "number"},
                        },
                    },
                    "description": "Data points with labels and values",
                },
                "title": {
                    "type": "string",
                    "description": "Chart title",
                },
                "width": {
                    "type": "integer",
                    "description": "Chart width in characters (default: 50)",
                },
            },
            "required": ["chart_type", "data", "title"],
        },
    },
    {
        "name": "detect_anomalies",
        "description": "Detect unusual patterns or outliers in the data using statistical methods.",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Table to analyze",
                },
                "column": {
                    "type": "string",
                    "description": "Column to check for anomalies",
                },
                "method": {
                    "type": "string",
                    "description": "Detection method",
                    "enum": ["zscore", "iqr"],
                    "default": "zscore",
                },
                "threshold": {
                    "type": "number",
                    "description": "Threshold for anomaly detection (default: 2.0 for zscore, 1.5 for IQR)",
                },
            },
            "required": ["table", "column"],
        },
    },
    {
        "name": "generate_insight",
        "description": "Generate a natural language insight or recommendation based on analysis results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "finding": {
                    "type": "string",
                    "description": "The analytical finding to explain",
                },
                "context": {
                    "type": "string",
                    "description": "Business context for the insight",
                },
                "insight_type": {
                    "type": "string",
                    "description": "Type of insight to generate",
                    "enum": ["explanation", "recommendation", "warning", "opportunity"],
                },
            },
            "required": ["finding", "insight_type"],
        },
    },
]


# =============================================================================
# Tool Implementations
# =============================================================================


def query_database(
    table: str,
    columns: list[str] | None = None,
    where: dict[str, Any] | None = None,
    group_by: str | None = None,
    order_by: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Execute a query against the simulated database.

    This implements a simple SQL-like query engine for demonstration.
    In production, use SQLAlchemy or direct database connections.
    """
    # Get the base data
    if table not in SAMPLE_DATA:
        return {
            "error": f"Unknown table: {table}. Available: {list(SAMPLE_DATA.keys())}"
        }

    data = SAMPLE_DATA[table].copy()

    # Apply WHERE filters
    if where:
        filtered = []
        for row in data:
            match = True
            for key, value in where.items():
                # Handle comparison operators
                if key.endswith("__gt"):
                    col = key[:-4]
                    if col in row and not (row[col] > value):
                        match = False
                elif key.endswith("__lt"):
                    col = key[:-4]
                    if col in row and not (row[col] < value):
                        match = False
                elif key.endswith("__gte"):
                    col = key[:-5]
                    if col in row and not (row[col] >= value):
                        match = False
                elif key.endswith("__lte"):
                    col = key[:-5]
                    if col in row and not (row[col] <= value):
                        match = False
                elif key.endswith("__contains"):
                    col = key[:-10]
                    if col in row and value.lower() not in str(row[col]).lower():
                        match = False
                else:
                    # Exact match
                    if key in row and row[key] != value:
                        match = False
            if match:
                filtered.append(row)
        data = filtered

    # Apply GROUP BY with aggregation
    if group_by:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in data:
            key = str(row.get(group_by, "Unknown"))
            if key not in groups:
                groups[key] = []
            groups[key].append(row)

        # Aggregate: count and sum numeric columns
        aggregated = []
        for group_key, rows in groups.items():
            agg_row = {group_by: group_key, "count": len(rows)}
            # Sum numeric columns
            for col in rows[0].keys():
                if isinstance(rows[0].get(col), (int, float)) and col not in [
                    "id",
                    "customer_id",
                    "product_id",
                ]:
                    agg_row[f"sum_{col}"] = round(sum(r.get(col, 0) for r in rows), 2)
                    agg_row[f"avg_{col}"] = round(
                        sum(r.get(col, 0) for r in rows) / len(rows), 2
                    )
            aggregated.append(agg_row)
        data = aggregated

    # Apply ORDER BY
    if order_by:
        descending = order_by.startswith("-")
        col = order_by[1:] if descending else order_by
        data = sorted(data, key=lambda x: x.get(col, 0), reverse=descending)

    # Apply LIMIT
    if limit:
        data = data[:limit]

    # Select columns
    if columns and columns != ["*"]:
        data = [{k: v for k, v in row.items() if k in columns} for row in data]

    return {
        "table": table,
        "row_count": len(data),
        "rows": data,
    }


def calculate_statistics(
    table: str,
    column: str,
    group_by: str | None = None,
) -> dict[str, Any]:
    """Calculate statistical measures for a column."""
    if table not in SAMPLE_DATA:
        return {"error": f"Unknown table: {table}"}

    data = SAMPLE_DATA[table]

    def calc_stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"error": "No values to analyze"}

        n = len(values)
        mean = sum(values) / n
        sorted_vals = sorted(values)
        median = (
            sorted_vals[n // 2]
            if n % 2
            else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        )
        variance = sum((x - mean) ** 2 for x in values) / n
        std_dev = variance**0.5

        return {
            "count": n,
            "sum": round(sum(values), 2),
            "mean": round(mean, 2),
            "median": round(median, 2),
            "std_dev": round(std_dev, 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    if group_by:
        # Group statistics
        groups: dict[str, list[float]] = {}
        for row in data:
            key = str(row.get(group_by, "Unknown"))
            if key not in groups:
                groups[key] = []
            if column in row and isinstance(row[column], (int, float)):
                groups[key].append(float(row[column]))

        return {
            "table": table,
            "column": column,
            "grouped_by": group_by,
            "groups": {k: calc_stats(v) for k, v in groups.items()},
        }
    else:
        # Overall statistics
        values = [
            float(row[column])
            for row in data
            if column in row and isinstance(row[column], (int, float))
        ]
        return {
            "table": table,
            "column": column,
            "statistics": calc_stats(values),
        }


def create_chart(
    chart_type: str,
    data: list[dict[str, Any]],
    title: str,
    width: int = 50,
) -> dict[str, Any]:
    """Create a text-based chart visualization."""
    if not data:
        return {"error": "No data provided for chart"}

    # Find max value for scaling
    max_value = max(d["value"] for d in data)
    if max_value == 0:
        max_value = 1

    lines = [f"\n{title}", "=" * len(title)]

    if chart_type == "bar":
        # Horizontal bar chart
        max_label_len = max(len(str(d["label"])) for d in data)
        for d in data:
            bar_length = int((d["value"] / max_value) * (width - max_label_len - 10))
            bar = "█" * bar_length
            label = str(d["label"]).ljust(max_label_len)
            lines.append(f"{label} | {bar} {d['value']:.1f}")

    elif chart_type == "line":
        # Simple line chart using ASCII
        height = 10
        for d in data:
            d["scaled"] = int((d["value"] / max_value) * height)

        for level in range(height, -1, -1):
            row = ""
            for d in data:
                if d["scaled"] >= level:
                    row += "●"
                else:
                    row += " "
            lines.append(f"{level:3d} |{row}")
        lines.append("    +" + "-" * len(data))
        lines.append("     " + "".join(str(d["label"])[0] for d in data))

    elif chart_type == "histogram":
        # Vertical histogram
        for d in data:
            bar_height = int((d["value"] / max_value) * 10)
            lines.append(f"{d['label']}: {'▇' * bar_height} ({d['value']:.0f})")

    chart_text = "\n".join(lines)
    return {
        "chart_type": chart_type,
        "title": title,
        "data_points": len(data),
        "chart": chart_text,
    }


def detect_anomalies(
    table: str,
    column: str,
    method: str = "zscore",
    threshold: float | None = None,
) -> dict[str, Any]:
    """Detect anomalies in the data."""
    if table not in SAMPLE_DATA:
        return {"error": f"Unknown table: {table}"}

    data = SAMPLE_DATA[table]
    values = [
        (i, row[column])
        for i, row in enumerate(data)
        if column in row and isinstance(row[column], (int, float))
    ]

    if not values:
        return {"error": f"No numeric values in column: {column}"}

    nums = [v[1] for v in values]
    mean = sum(nums) / len(nums)
    std_dev = (sum((x - mean) ** 2 for x in nums) / len(nums)) ** 0.5

    anomalies = []

    if method == "zscore":
        thresh = threshold or 2.0
        if std_dev > 0:
            for idx, val in values:
                zscore = abs(val - mean) / std_dev
                if zscore > thresh:
                    anomalies.append(
                        {
                            "row_index": idx,
                            "value": val,
                            "zscore": round(zscore, 2),
                            "row": data[idx],
                        }
                    )
    elif method == "iqr":
        thresh = threshold or 1.5
        sorted_nums = sorted(nums)
        q1 = sorted_nums[len(sorted_nums) // 4]
        q3 = sorted_nums[3 * len(sorted_nums) // 4]
        iqr = q3 - q1
        lower = q1 - thresh * iqr
        upper = q3 + thresh * iqr

        for idx, val in values:
            if val < lower or val > upper:
                anomalies.append(
                    {
                        "row_index": idx,
                        "value": val,
                        "bounds": [round(lower, 2), round(upper, 2)],
                        "row": data[idx],
                    }
                )

    return {
        "table": table,
        "column": column,
        "method": method,
        "threshold": threshold or (2.0 if method == "zscore" else 1.5),
        "total_rows": len(values),
        "anomalies_found": len(anomalies),
        "anomalies": anomalies[:10],  # Limit output
    }


def generate_insight(
    finding: str,
    insight_type: str,
    context: str | None = None,
) -> dict[str, Any]:
    """Generate a business insight based on findings."""
    templates = {
        "explanation": f"**Analysis Finding**: {finding}\n\n"
        f"This indicates that {finding.lower()}. "
        f"{'Given the context of ' + context + ', this ' if context else 'This '}"
        f"suggests patterns in the underlying data that warrant attention.",
        "recommendation": f"**Recommendation based on: {finding}**\n\n"
        f"Based on this finding, we recommend:\n"
        f"1. Investigate the root causes of this pattern\n"
        f"2. Consider adjustments to current strategies\n"
        f"3. Monitor this metric closely going forward\n"
        f"{'Context: ' + context if context else ''}",
        "warning": f"⚠️ **Warning**: {finding}\n\n"
        f"This finding requires immediate attention. "
        f"{'In the context of ' + context + ', ' if context else ''}"
        f"Failure to address this could impact business outcomes.",
        "opportunity": f"💡 **Opportunity Identified**: {finding}\n\n"
        f"This represents a potential opportunity for improvement. "
        f"{'Given ' + context + ', ' if context else ''}"
        f"Consider allocating resources to capitalize on this finding.",
    }

    return {
        "insight_type": insight_type,
        "finding": finding,
        "context": context,
        "insight": templates.get(insight_type, templates["explanation"]),
    }


# =============================================================================
# Agent Setup
# =============================================================================


def create_data_analyst() -> tuple[Any, Any]:
    """Create the data analyst agent."""
    handlers = {
        "query_database": query_database,
        "calculate_statistics": calculate_statistics,
        "create_chart": create_chart,
        "detect_anomalies": detect_anomalies,
        "generate_insight": generate_insight,
    }

    # Ingest Anthropic format tools
    funcs, defs = from_anthropic_tools(ANTHROPIC_TOOLS, handlers=handlers)

    print(f"✓ Ingested {len(defs)} tools from Anthropic format:")
    for d in defs:
        print(f"  • {d.name}: {d.description[:50]}...")

    # Create sandbox with constraints
    sandbox = create_sandbox_tool(
        tools=funcs,
        max_calls={
            "query_database": 20,
            "calculate_statistics": 15,
            "create_chart": 10,
            "detect_anomalies": 10,
            "generate_insight": 20,
        },
    )

    client = OpenAI()
    return sandbox, client


def run_agent(sandbox: Any, client: Any, analysis_request: str) -> str:
    """Run the data analyst agent."""

    system_prompt = """You are a data analyst. Write CONCISE JavaScript (under 40 lines).

Tools (use await):
- query_database({table, group_by?, order_by?, limit?}) - tables: orders, products, customers
- calculate_statistics({table, column, group_by?})
- create_chart({chart_type, data, title}) - types: bar, line
- detect_anomalies({table, column})
- generate_insight({finding, insight_type}) - types: recommendation, opportunity

Use console.log() for ALL output. Keep code SHORT - focus on key insights only."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": analysis_request},
        ],
        temperature=0.7,
    )

    code = response.choices[0].message.content
    code_match = re.search(r"```(?:javascript|js)?\n(.*?)```", code, re.DOTALL)
    if code_match:
        code = code_match.group(1)

    print("\n📝 Agent generated analysis code:")
    print("-" * 40)
    print(code[:600] + "..." if len(code) > 600 else code)
    print("-" * 40)

    print("\n⚡ Executing analysis...")
    return sandbox.run(code, language="javascript")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run the data analyst demo."""
    print("=" * 60)
    print("Data Analyst Agent")
    print("Anthropic Format Tool Ingestion Demo")
    print("=" * 60)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY not set")
        return

    print("\n🔧 Setting up agent...")
    sandbox, client = create_data_analyst()

    # Show sample data info
    print("\n📊 Sample data loaded:")
    print(f"  • {len(SAMPLE_DATA['products'])} products")
    print(f"  • {len(SAMPLE_DATA['orders'])} orders (last 90 days)")
    print(f"  • {len(SAMPLE_DATA['customers'])} customers")

    analysis_request = """Analyze sales: show top categories, sales by region with a chart, and one key insight."""

    print("\n📋 Analysis request:")
    print(analysis_request)
    print("\n" + "=" * 60)

    result = run_agent(sandbox, client, analysis_request)

    print("\n📊 Analysis Results:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
