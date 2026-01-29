#!/usr/bin/env python3
"""Code Review Assistant - LangChain Format Tool Ingestion Demo.

This agent helps developers review and improve their code by:
1. Analyzing code for potential issues (bugs, performance)
2. Checking code style and best practices
3. Suggesting improvements and refactoring opportunities
4. Generating documentation and tests

This demonstrates ingesting LangChain @tool decorated functions into
amla-sandbox, showing how existing LangChain tools can be reused with
Amla's secure sandbox execution.

Real-world use case: Automated code review for CI/CD pipelines, helping
developers catch issues before code review, and generating documentation.

Requirements:
    pip install amla-sandbox openai python-dotenv langchain-core

Usage:
    export AMLA_WASM_PATH=/path/to/amla_sandbox.wasm
    python code_reviewer.py
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import tool
from openai import OpenAI

from amla_sandbox import create_sandbox_tool
from amla_sandbox.tools import from_langchain

# =============================================================================
# Configuration
# =============================================================================

env_path = Path(__file__).parent
while env_path != env_path.parent:
    if (env_path / ".env").exists():
        load_dotenv(env_path / ".env")
        break
    env_path = env_path.parent

if "AMLA_WASM_PATH" not in os.environ:
    wasm_path = Path(__file__).parents[5] / "rust/amla-sandbox/pkg/amla_sandbox_bg.wasm"
    if wasm_path.exists():
        os.environ["AMLA_WASM_PATH"] = str(wasm_path)


# =============================================================================
# Sample Code for Review
# =============================================================================

# Sample Python code with various issues for the agent to find
SAMPLE_CODE = """
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    return {"id": user_id}

def divide(a, b):
    return a / b

def delete_all(users):
    for user in users:
        users.remove(user)

class Processor:
    def process(self):
        try:
            return self.run()
        except:
            return None
"""


# =============================================================================
# LangChain Tool Definitions
# =============================================================================


@tool
def analyze_code(code: str, analysis_type: str = "all") -> dict[str, Any]:
    """Analyze Python code for potential issues.

    Args:
        code: The Python source code to analyze.
        analysis_type: Type of analysis - 'security', 'bugs', 'style', 'performance', or 'all'.

    Returns:
        Dictionary containing found issues categorized by type.
    """
    issues: dict[str, list[dict[str, Any]]] = {
        "security": [],
        "bugs": [],
        "style": [],
        "performance": [],
    }

    lines = code.split("\n")

    # Security checks
    if analysis_type in ("security", "all"):
        security_patterns = [
            (
                r"SELECT.*\+.*str\(",
                "Possible SQL injection via string concatenation",
                "critical",
            ),
            (
                r"\.format\([^)]*\).*(?:SELECT|INSERT|UPDATE)",
                "Possible SQL injection via format",
                "high",
            ),
        ]
        for line_num, line in enumerate(lines, 1):
            for pattern, message, severity in security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues["security"].append(
                        {
                            "line": line_num,
                            "message": message,
                            "severity": severity,
                            "code": line.strip(),
                        }
                    )

    # Bug detection
    if analysis_type in ("bugs", "all"):
        for line_num, line in enumerate(lines, 1):
            # Bare except
            if re.match(r"\s*except\s*:", line):
                issues["bugs"].append(
                    {
                        "line": line_num,
                        "message": "Bare except catches all exceptions including KeyboardInterrupt",
                        "severity": "medium",
                        "code": line.strip(),
                    }
                )

            # Modifying collection while iterating
            if re.search(r"for\s+\w+\s+in\s+self\.(\w+):", line):
                match = re.search(r"for\s+\w+\s+in\s+self\.(\w+):", line)
                if match:
                    coll = match.group(1)
                    for j in range(line_num, min(line_num + 3, len(lines))):
                        if f"self.{coll}.remove" in lines[j]:
                            issues["bugs"].append(
                                {
                                    "line": line_num,
                                    "message": "Modifying collection while iterating",
                                    "severity": "high",
                                    "code": line.strip(),
                                }
                            )
                            break

            # Division without zero check
            if re.search(r"return\s+\w+\s*/\s*\w+", line):
                preceding = "\n".join(lines[max(0, line_num - 4) : line_num])
                if (
                    "if" not in preceding
                    and "== 0" not in preceding
                    and "!= 0" not in preceding
                ):
                    issues["bugs"].append(
                        {
                            "line": line_num,
                            "message": "Division without zero check",
                            "severity": "medium",
                            "code": line.strip(),
                        }
                    )

    # Style issues
    if analysis_type in ("style", "all"):
        for line_num, line in enumerate(lines, 1):
            # Missing type hints
            if re.match(r"\s*def\s+\w+\([^)]*\)\s*:", line):
                if "->" not in line and "self" in line:
                    issues["style"].append(
                        {
                            "line": line_num,
                            "message": "Method missing return type annotation",
                            "severity": "low",
                            "code": line.strip(),
                        }
                    )

            # Magic numbers
            if re.search(r"\b\d+\.\d{2,}\b", line) and "def " not in line:
                issues["style"].append(
                    {
                        "line": line_num,
                        "message": "Magic number - consider named constant",
                        "severity": "low",
                        "code": line.strip(),
                    }
                )

            # Print instead of logging
            if "print(" in line and (
                "error" in line.lower() or "warning" in line.lower()
            ):
                issues["style"].append(
                    {
                        "line": line_num,
                        "message": "Use logging module instead of print for errors",
                        "severity": "low",
                        "code": line.strip(),
                    }
                )

    # Performance issues
    if analysis_type in ("performance", "all"):
        for line_num, line in enumerate(lines, 1):
            # Appending in loop - could be list comprehension
            if ".append(" in line:
                for j in range(max(0, line_num - 3), line_num):
                    if "for " in lines[j]:
                        issues["performance"].append(
                            {
                                "line": line_num,
                                "message": "Consider list comprehension instead of append in loop",
                                "severity": "low",
                                "code": line.strip(),
                            }
                        )
                        break

    total = sum(len(v) for v in issues.values())
    critical = sum(
        1 for cat in issues.values() for i in cat if i.get("severity") == "critical"
    )
    high = sum(1 for cat in issues.values() for i in cat if i.get("severity") == "high")

    return {
        "total_issues": total,
        "critical_count": critical,
        "high_count": high,
        "issues": issues,
        "summary": f"Found {total} issues: {critical} critical, {high} high severity",
    }


@tool
def check_complexity(code: str) -> dict[str, Any]:
    """Analyze code complexity metrics.

    Args:
        code: Python source code to analyze.

    Returns:
        Complexity metrics including cyclomatic complexity estimates.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}

    functions: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexity = 1
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.While, ast.For)):
                    complexity += 1
                elif isinstance(child, ast.BoolOp):
                    complexity += len(child.values) - 1
                elif isinstance(child, ast.ExceptHandler):
                    complexity += 1

            end_line = getattr(node, "end_lineno", None)
            lines = (end_line - node.lineno + 1) if end_line else 0
            functions.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "complexity": complexity,
                    "lines": lines,
                    "rating": "good"
                    if complexity < 5
                    else "moderate"
                    if complexity < 10
                    else "high",
                }
            )

    functions.sort(key=lambda x: x["complexity"], reverse=True)
    avg = sum(f["complexity"] for f in functions) / len(functions) if functions else 0

    return {
        "function_count": len(functions),
        "average_complexity": round(avg, 2),
        "functions": functions,
        "recommendation": (
            "Code complexity is manageable."
            if avg < 5
            else "Consider refactoring complex functions."
            if avg < 10
            else "High complexity - refactoring recommended."
        ),
    }


@tool
def suggest_fixes(code: str, issue_type: str = "all") -> dict[str, Any]:
    """Suggest fixes for common code issues.

    Args:
        code: The problematic code snippet.
        issue_type: Type of issue - 'security', 'performance', 'style', or 'all'.

    Returns:
        Suggested fix with explanation.
    """
    all_suggestions = {
        "security": {
            "sql_injection": {
                "issue": "String concatenation in SQL queries",
                "fix": "Use parameterized queries with placeholders",
                "example": 'query = "SELECT * FROM users WHERE id = ?"; cursor.execute(query, (user_id,))',
            },
        },
        "bugs": {
            "bare_except": {
                "issue": "Bare except clause catches everything",
                "fix": "Catch specific exceptions",
                "example": "except ValueError as e: handle_error(e)",
            },
            "division_zero": {
                "issue": "Division without zero check",
                "fix": "Check denominator before division",
                "example": "if b == 0: raise ValueError('Division by zero'); return a / b",
            },
        },
        "style": {
            "type_hints": {
                "issue": "Missing type annotations",
                "fix": "Add type hints for clarity",
                "example": "def add_user(self, user: dict[str, Any]) -> None:",
            },
        },
        "performance": {
            "list_comprehension": {
                "issue": "Append in loop",
                "fix": "Use list comprehension for cleaner code",
                "example": "result = [item * 2 for item in items if item > 0]",
            },
        },
    }

    # Flatten suggestions based on issue_type filter
    suggestions = {}
    if issue_type == "all":
        for category in all_suggestions.values():
            suggestions.update(category)
    elif issue_type in all_suggestions:
        suggestions = all_suggestions[issue_type]

    code_lower = code.lower()
    matched = []

    if "sql_injection" in suggestions and "select" in code_lower and "+" in code_lower:
        matched.append(suggestions["sql_injection"])
    if "bare_except" in suggestions and "except:" in code_lower:
        matched.append(suggestions["bare_except"])
    if "division_zero" in suggestions and "/" in code and "if " not in code:
        matched.append(suggestions["division_zero"])
    if "type_hints" in suggestions and "def " in code and "->" not in code:
        matched.append(suggestions["type_hints"])
    if "list_comprehension" in suggestions and ".append(" in code and "for " in code:
        matched.append(suggestions["list_comprehension"])

    return (
        {"suggestions_count": len(matched), "suggestions": matched}
        if matched
        else {
            "message": "No specific suggestions found.",
            "advice": "Review for edge cases and documentation.",
        }
    )


@tool
def generate_docstring(code: str, style: str = "google") -> dict[str, Any]:
    """Generate docstrings for Python functions.

    Args:
        code: Python function code to document.
        style: Docstring style - 'google', 'numpy', or 'sphinx'.

    Returns:
        Generated docstring following the specified style.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"Cannot parse: {e}"}

    docstrings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [
                {
                    "name": a.arg,
                    "type": ast.unparse(a.annotation) if a.annotation else "Any",
                }
                for a in node.args.args
            ]
            ret = ast.unparse(node.returns) if node.returns else "None"

            if style == "google":
                lines = [f'"""Brief description of {node.name}.', ""]
                if args:
                    lines.append("Args:")
                    for a in args:
                        if a["name"] != "self":
                            lines.append(f"    {a['name']}: Description.")
                if ret != "None":
                    lines.extend(["", "Returns:", f"    {ret}: Description."])
                lines.append('"""')
            else:
                lines = [f'"""Description of {node.name}."""']

            docstrings.append({"function": node.name, "docstring": "\n".join(lines)})

    return {"style": style, "count": len(docstrings), "docstrings": docstrings}


@tool
def generate_tests(code: str, framework: str = "pytest") -> dict[str, Any]:
    """Generate unit test stubs for Python functions.

    Args:
        code: Python source code containing functions to test.
        framework: Test framework - 'pytest' or 'unittest'.

    Returns:
        Generated test code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"Cannot parse: {e}"}

    tests = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            name = node.name
            test = f"def test_{name}():\n    # TODO: Implement\n    pass"
            tests.append({"function": name, "test": test})

    return {"framework": framework, "count": len(tests), "tests": tests}


# =============================================================================
# Agent Setup
# =============================================================================


def create_code_reviewer() -> tuple[Any, Any]:
    """Create the code review assistant agent."""
    langchain_tools = [
        analyze_code,
        check_complexity,
        suggest_fixes,
        generate_docstring,
        generate_tests,
    ]
    funcs, defs = from_langchain(langchain_tools)

    print(f"✓ Ingested {len(defs)} tools from LangChain @tool format:")
    for d in defs:
        print(f"  • {d.name}: {d.description[:50]}...")

    sandbox = create_sandbox_tool(tools=funcs, max_calls=20)
    client = OpenAI()
    return sandbox, client


def run_agent(sandbox: Any, client: Any, code: str, request: str) -> str:
    """Run the code review agent."""
    escaped = code.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    system_prompt = f"""Write JavaScript. The code to review is already defined:
const codeToReview = `{escaped}`;

Tools (use await):
- analyze_code({{code, analysis_type}}) - returns {{issues: {{security:[], bugs:[]}}, summary}}
- suggest_fixes({{code, issue_type}}) - returns {{suggestions}}

Write 10 lines max. Use console.log() for output. No markdown."""

    prompt = request

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    js = response.choices[0].message.content
    match = re.search(r"```(?:javascript|js)?\n(.*?)```", js, re.DOTALL)
    js = match.group(1) if match else js

    # Prepend the code definition
    full_js = f"const codeToReview = `{escaped}`;\n{js}"

    print("\n📝 Generated review code:")
    print("-" * 40)
    print(js[:400] + "..." if len(js) > 400 else js)
    print("-" * 40)
    print("\n⚡ Executing...")

    return sandbox.run(full_js, language="javascript")


def main():
    """Run the code reviewer demo."""
    print("=" * 60)
    print("Code Review Assistant - LangChain Format Demo")
    print("=" * 60)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ OPENAI_API_KEY not set")
        return

    sandbox, client = create_code_reviewer()
    print(f"\n📄 Reviewing {len(SAMPLE_CODE.split(chr(10)))} lines of sample code...")

    result = run_agent(
        sandbox,
        client,
        SAMPLE_CODE,
        "Analyze for security and bugs, show top 3 issues with fixes.",
    )

    print("\n📊 Results:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
