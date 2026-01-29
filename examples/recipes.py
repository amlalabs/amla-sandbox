#!/usr/bin/env python3
"""Recipes & Quick Solutions - Common Patterns for amla-sandbox

This cookbook contains copy-paste recipes for common tasks.
Each recipe is self-contained and can be run independently.

Prerequisites:
    pip install amla-sandbox

Run:
    python recipes.py

Jump to a specific recipe:
    python -c "from recipes import recipe_json_processing; recipe_json_processing()"
"""

from __future__ import annotations

import json
from typing import Any

from amla_sandbox import (
    MethodCapability,
    Sandbox,
    SandboxTool,
    create_sandbox_tool,
)


# =============================================================================
# RECIPE: Quick Shell Commands
# =============================================================================


def recipe_quick_shell() -> None:
    """Run shell commands in the sandbox."""
    print("\n" + "=" * 60)
    print("Recipe: Quick Shell Commands")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Simple command
    result = sandbox.run("echo 'Hello World'", language="shell")
    print(f"Echo: {result}")

    # Pipeline
    result = sandbox.run(
        "echo 'apple banana cherry' | tr ' ' '\\n' | sort", language="shell"
    )
    print(f"Sorted: {result}")

    # Multiple commands
    result = sandbox.run(
        """
        echo 'Line 1' > /workspace/test.txt
        echo 'Line 2' >> /workspace/test.txt
        cat /workspace/test.txt
    """,
        language="shell",
    )
    print(f"Multi-command: {result}")


# =============================================================================
# RECIPE: JSON Processing with jq
# =============================================================================


def recipe_json_processing() -> None:
    """Process JSON data with jq."""
    print("\n" + "=" * 60)
    print("Recipe: JSON Processing with jq")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Write JSON data
    data = json.dumps(
        {
            "users": [
                {"name": "Alice", "age": 30, "role": "admin"},
                {"name": "Bob", "age": 25, "role": "user"},
                {"name": "Carol", "age": 35, "role": "admin"},
            ]
        }
    )

    # Extract with jq
    result = sandbox.run(
        f"""
        echo '{data}' | jq '.users[] | select(.role == "admin") | .name'
    """,
        language="shell",
    )
    print(f"Admins: {result}")

    # Transform JSON
    result = sandbox.run(
        f"""
        echo '{data}' | jq '[.users[] | {{username: .name, isAdmin: (.role == "admin")}}]'
    """,
        language="shell",
    )
    print(f"Transformed: {result}")


# =============================================================================
# RECIPE: Log Analysis
# =============================================================================


def recipe_log_analysis() -> None:
    """Analyze log files with grep, cut, sort, uniq."""
    print("\n" + "=" * 60)
    print("Recipe: Log Analysis")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Create sample log
    sandbox.run(
        r"""
        cat > /workspace/app.log << 'EOF'
2024-01-15 10:00:01 INFO User login: alice
2024-01-15 10:00:05 ERROR Database connection failed
2024-01-15 10:00:10 INFO User login: bob
2024-01-15 10:00:15 WARN High memory usage
2024-01-15 10:00:20 ERROR API timeout
2024-01-15 10:00:25 INFO User logout: alice
EOF
    """,
        language="shell",
    )

    # Count by log level
    result = sandbox.run(
        """
        cat /workspace/app.log | cut -d' ' -f3 | sort | uniq -c | sort -rn
    """,
        language="shell",
    )
    print(f"Log levels:\n{result}")

    # Find errors
    result = sandbox.run("grep ERROR /workspace/app.log", language="shell")
    print(f"Errors:\n{result}")


# =============================================================================
# RECIPE: CSV Processing
# =============================================================================


def recipe_csv_processing() -> None:
    """Process CSV files with JavaScript."""
    print("\n" + "=" * 60)
    print("Recipe: CSV Processing")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Create CSV
    sandbox.run(
        """
        cat > /workspace/sales.csv << 'EOF'
date,product,quantity,price
2024-01-01,Widget,10,25.00
2024-01-02,Gadget,5,50.00
2024-01-03,Widget,15,25.00
2024-01-04,Gadget,8,50.00
2024-01-05,Widget,20,25.00
EOF
    """,
        language="shell",
    )

    # Sum by product using JavaScript (CSV processing is easier in JS)
    result = sandbox.run(
        """
        const csv = await fs.readFile('/workspace/sales.csv');
        const lines = csv.trim().split('\\n').slice(1);  // Skip header

        const byProduct = {};
        for (const line of lines) {
            const [date, product, qty, price] = line.split(',');
            byProduct[product] = (byProduct[product] || 0) + parseFloat(qty) * parseFloat(price);
        }

        for (const [product, total] of Object.entries(byProduct)) {
            console.log(`${product}: $${total.toFixed(2)}`);
        }
    """,
        language="javascript",
    )
    print(f"Sales by product:\n{result}")

    # Total revenue
    result = sandbox.run(
        """
        const csv = await fs.readFile('/workspace/sales.csv');
        const lines = csv.trim().split('\\n').slice(1);

        let total = 0;
        for (const line of lines) {
            const parts = line.split(',');
            total += parseFloat(parts[2]) * parseFloat(parts[3]);
        }
        console.log('$' + total.toFixed(2));
    """,
        language="javascript",
    )
    print(f"Total revenue: {result}")


# =============================================================================
# RECIPE: Tool with Constraints
# =============================================================================


def recipe_constrained_tool() -> None:
    """Create a tool with parameter constraints."""
    print("\n" + "=" * 60)
    print("Recipe: Constrained Tool")
    print("=" * 60)

    def transfer_money(
        from_account: str, to_account: str, amount: float
    ) -> dict[str, Any]:
        """Transfer money between accounts."""
        return {
            "status": "success",
            "from": from_account,
            "to": to_account,
            "amount": amount,
            "message": f"Transferred ${amount:.2f}",
        }

    # Create tool with constraints
    sandbox = create_sandbox_tool(
        tools=[transfer_money],
        constraints={
            "transfer_money": {
                "amount": "<=1000",  # Max $1000 per transfer
            }
        },
    )

    # Valid transfer
    result = sandbox.run(
        """
        const result = await transfer_money({
            from_account: "checking",
            to_account: "savings",
            amount: 500
        });
        console.log(JSON.stringify(result));
    """,
        language="javascript",
    )
    print(f"Transfer: {result}")


# =============================================================================
# RECIPE: Multiple Tools
# =============================================================================


def recipe_multiple_tools() -> None:
    """Register multiple tools and compose them."""
    print("\n" + "=" * 60)
    print("Recipe: Multiple Tools")
    print("=" * 60)

    def get_user(user_id: str) -> dict[str, Any]:
        """Get user details."""
        users = {
            "u1": {"name": "Alice", "email": "alice@example.com"},
            "u2": {"name": "Bob", "email": "bob@example.com"},
        }
        return users.get(user_id, {"error": "not found"})

    def send_email(to: str, subject: str, body: str) -> dict[str, Any]:
        """Send an email."""
        return {"sent": True, "to": to, "subject": subject, "bodyLength": len(body)}

    sandbox = create_sandbox_tool(tools=[get_user, send_email])

    # Compose tools
    result = sandbox.run(
        """
        // Get user and send them an email
        const user = await get_user({user_id: "u1"});
        if (user.email) {
            const email = await send_email({
                to: user.email,
                subject: "Hello",
                body: "Welcome!"
            });
            console.log(JSON.stringify({user, email}));
        }
    """,
        language="javascript",
    )
    print(f"Result: {result}")


# =============================================================================
# RECIPE: File Operations with VFS
# =============================================================================


def recipe_vfs_operations() -> None:
    """Use the virtual filesystem for data persistence."""
    print("\n" + "=" * 60)
    print("Recipe: VFS Operations")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Write and read JSON
    result = sandbox.run(
        """
        // Write JSON
        const data = {items: [1, 2, 3], total: 6};
        await fs.writeFile('/workspace/data.json', JSON.stringify(data, null, 2));

        // Read it back
        const content = await fs.readFile('/workspace/data.json');
        const parsed = JSON.parse(content);
        console.log('Total:', parsed.total);
    """,
        language="javascript",
    )
    print(f"VFS result: {result}")

    # Directory operations
    result = sandbox.run(
        """
        await fs.mkdir('/workspace/reports', {recursive: true});
        await fs.writeFile('/workspace/reports/q1.txt', 'Q1 Report');
        await fs.writeFile('/workspace/reports/q2.txt', 'Q2 Report');
        const files = await fs.readdir('/workspace/reports');
        console.log('Files:', files.join(', '));
    """,
        language="javascript",
    )
    print(f"Directory: {result}")


# =============================================================================
# RECIPE: Data Transformation Pipeline
# =============================================================================


def recipe_data_pipeline() -> None:
    """Build a data transformation pipeline."""
    print("\n" + "=" * 60)
    print("Recipe: Data Transformation Pipeline")
    print("=" * 60)

    def fetch_orders() -> list[dict[str, Any]]:
        """Fetch orders from database."""
        return [
            {"id": 1, "customer": "Alice", "total": 150.00, "status": "completed"},
            {"id": 2, "customer": "Bob", "total": 75.50, "status": "pending"},
            {"id": 3, "customer": "Carol", "total": 200.00, "status": "completed"},
            {"id": 4, "customer": "Alice", "total": 50.00, "status": "completed"},
        ]

    sandbox = create_sandbox_tool(tools=[fetch_orders])

    result = sandbox.run(
        """
        // Fetch data
        const orders = await fetch_orders({});

        // Transform: filter completed, group by customer, calculate totals
        const completed = orders.filter(o => o.status === 'completed');

        const byCustomer = {};
        for (const order of completed) {
            if (!byCustomer[order.customer]) {
                byCustomer[order.customer] = {count: 0, total: 0};
            }
            byCustomer[order.customer].count++;
            byCustomer[order.customer].total += order.total;
        }

        console.log(JSON.stringify(byCustomer, null, 2));
    """,
        language="javascript",
    )
    print(f"Customer totals:\n{result}")


# =============================================================================
# RECIPE: Error Handling
# =============================================================================


def recipe_error_handling() -> None:
    """Handle errors gracefully in sandbox code."""
    print("\n" + "=" * 60)
    print("Recipe: Error Handling")
    print("=" * 60)

    def risky_operation(should_fail: bool) -> dict[str, Any]:
        """An operation that might fail."""
        if should_fail:
            raise ValueError("Operation failed!")
        return {"status": "success"}

    sandbox = create_sandbox_tool(tools=[risky_operation])

    result = sandbox.run(
        """
        // Try-catch for error handling
        let results = [];

        try {
            const success = await risky_operation({should_fail: false});
            results.push({op: 'success', result: success});
        } catch (e) {
            results.push({op: 'success', error: e.message});
        }

        try {
            const failure = await risky_operation({should_fail: true});
            results.push({op: 'failure', result: failure});
        } catch (e) {
            results.push({op: 'failure', error: e.message});
        }

        console.log(JSON.stringify(results, null, 2));
    """,
        language="javascript",
    )
    print(f"Results:\n{result}")


# =============================================================================
# RECIPE: Streaming Output
# =============================================================================


def recipe_streaming() -> None:
    """Capture output from sandbox execution."""
    print("\n" + "=" * 60)
    print("Recipe: Capturing Output")
    print("=" * 60)

    with Sandbox(
        tools=[],
        capabilities=[MethodCapability(method_pattern="**")],
    ) as sandbox:
        # All console.log output is captured in the result
        result = sandbox.execute("""
            console.log('Starting...');
            for (let i = 1; i <= 3; i++) {
                console.log('Step ' + i + ' of 3');
            }
            console.log('Done!');
        """)
        print("Captured output:")
        for line in result.strip().split("\n"):
            print(f"  > {line}")


# =============================================================================
# RECIPE: Batch Processing
# =============================================================================


def recipe_batch_processing() -> None:
    """Process items in batches."""
    print("\n" + "=" * 60)
    print("Recipe: Batch Processing")
    print("=" * 60)

    def process_item(item_id: int) -> dict[str, Any]:
        """Process a single item."""
        return {"id": item_id, "processed": True, "result": item_id * 2}

    sandbox = create_sandbox_tool(tools=[process_item])

    result = sandbox.run(
        """
        // Process items in batches of 3
        const items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        const batchSize = 3;
        const results = [];

        for (let i = 0; i < items.length; i += batchSize) {
            const batch = items.slice(i, i + batchSize);
            console.log('Processing batch:', batch);

            for (const item of batch) {
                const result = await process_item({item_id: item});
                results.push(result);
            }
        }

        console.log('\\nFinal results:', results.length, 'items');
        console.log(JSON.stringify(results.slice(0, 3), null, 2));
    """,
        language="javascript",
    )
    print(f"Batch processing:\n{result}")


# =============================================================================
# RECIPE: Configuration Management
# =============================================================================


def recipe_config_management() -> None:
    """Manage configuration in the sandbox."""
    print("\n" + "=" * 60)
    print("Recipe: Configuration Management")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    result = sandbox.run(
        """
        // Write config
        const config = {
            database: {
                host: 'localhost',
                port: 5432,
                name: 'myapp'
            },
            features: {
                darkMode: true,
                notifications: false
            }
        };
        await fs.writeFile('/workspace/config.json', JSON.stringify(config, null, 2));

        // Read and merge with defaults
        const defaults = {
            database: { timeout: 30 },
            features: { darkMode: false, notifications: true, beta: false }
        };

        const saved = JSON.parse(await fs.readFile('/workspace/config.json'));

        // Deep merge
        const merged = {
            database: {...defaults.database, ...saved.database},
            features: {...defaults.features, ...saved.features}
        };

        console.log('Merged config:');
        console.log(JSON.stringify(merged, null, 2));
    """,
        language="javascript",
    )
    print(f"Config:\n{result}")


# =============================================================================
# RECIPE: Text Templating
# =============================================================================


def recipe_text_templating() -> None:
    """Generate text from templates."""
    print("\n" + "=" * 60)
    print("Recipe: Text Templating")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    result = sandbox.run(
        r"""
        // Simple template function
        function template(str, data) {
            return str.replace(/\{\{(\w+)\}\}/g, (_, key) => data[key] || '');
        }

        const tmpl = 'Hello {{name}}, your order #{{orderId}} is {{status}}.';

        const messages = [
            {name: 'Alice', orderId: '1001', status: 'shipped'},
            {name: 'Bob', orderId: '1002', status: 'processing'},
        ];

        for (const data of messages) {
            console.log(template(tmpl, data));
        }
    """,
        language="javascript",
    )
    print(f"Templates:\n{result}")


# =============================================================================
# RECIPE: Using SandboxTool for LangGraph
# =============================================================================


def recipe_langgraph_tool() -> None:
    """Create a tool for LangGraph integration."""
    print("\n" + "=" * 60)
    print("Recipe: LangGraph Tool")
    print("=" * 60)

    def search_database(query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search the database."""
        # Simulate search results (limit used to cap results)
        results = [
            {"id": 1, "title": f"Result for '{query}'", "score": 0.95},
            {"id": 2, "title": f"Another match for '{query}'", "score": 0.87},
            {"id": 3, "title": f"Third result for '{query}'", "score": 0.75},
        ]
        return results[:limit]

    # Create a SandboxTool (for LangGraph)
    sandbox_tool = SandboxTool.from_functions([search_database])

    # The tool can be used with LangGraph
    print("Created SandboxTool with functions:", [search_database.__name__])
    print(
        "Use with LangGraph: create_react_agent(model, [sandbox_tool.as_langchain_tool()])"
    )

    # Can also run directly
    result = sandbox_tool.run(
        """
        const results = await search_database({query: "python tutorials"});
        console.log(JSON.stringify(results, null, 2));
    """,
        language="javascript",
    )
    print(f"Direct run result:\n{result}")


# =============================================================================
# RECIPE: Reusing Sandbox State
# =============================================================================


def recipe_reuse_sandbox() -> None:
    """Reuse sandbox for multiple related operations."""
    print("\n" + "=" * 60)
    print("Recipe: Reusing Sandbox State")
    print("=" * 60)

    # Create sandbox once
    with Sandbox(
        tools=[],
        capabilities=[MethodCapability(method_pattern="**")],
    ) as sandbox:
        # First operation: setup
        sandbox.execute("""
            globalThis.counter = 0;
            globalThis.log = [];
        """)

        # Second operation: use shared state
        sandbox.execute("""
            globalThis.counter++;
            globalThis.log.push('incremented to ' + globalThis.counter);
        """)

        # Third operation: more updates
        sandbox.execute("""
            globalThis.counter += 5;
            globalThis.log.push('added 5, now ' + globalThis.counter);
        """)

        # Final operation: report
        result = sandbox.execute("""
            console.log('Final counter:', globalThis.counter);
            console.log('Log:', globalThis.log.join('; '));
        """)
        print(f"Result: {result}")


# =============================================================================
# RECIPE: Working with Dates
# =============================================================================


def recipe_date_handling() -> None:
    """Handle dates in JavaScript."""
    print("\n" + "=" * 60)
    print("Recipe: Date Handling")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    result = sandbox.run(
        """
        const now = new Date();
        const events = [
            {name: 'Meeting', date: new Date('2024-06-15T10:00:00')},
            {name: 'Deadline', date: new Date('2024-06-20T17:00:00')},
            {name: 'Review', date: new Date('2024-06-18T14:30:00')},
        ];

        // Sort by date
        events.sort((a, b) => a.date - b.date);

        console.log('Events by date:');
        for (const event of events) {
            const dateStr = event.date.toISOString().split('T')[0];
            console.log(`  ${dateStr}: ${event.name}`);
        }
    """,
        language="javascript",
    )
    print(f"Date handling:\n{result}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all recipes."""
    print("=" * 60)
    print("Recipes & Quick Solutions Cookbook")
    print("=" * 60)

    recipes = [
        recipe_quick_shell,
        recipe_json_processing,
        recipe_log_analysis,
        recipe_csv_processing,
        recipe_constrained_tool,
        recipe_multiple_tools,
        recipe_vfs_operations,
        recipe_data_pipeline,
        recipe_error_handling,
        recipe_streaming,
        recipe_batch_processing,
        recipe_config_management,
        recipe_text_templating,
        recipe_langgraph_tool,
        recipe_reuse_sandbox,
        recipe_date_handling,
    ]

    for recipe in recipes:
        try:
            recipe()
        except Exception as e:
            print(f"\n[ERROR in {recipe.__name__}]: {e}")

    print("\n" + "=" * 60)
    print("Cookbook Complete!")
    print("=" * 60)
    print("""
    Quick reference:

    Shell commands:     create_sandbox_tool().run("command")
    With tools:         create_sandbox_tool(tools=[fn]).run("await fn(...)")
    With constraints:   create_sandbox_tool(tools=[fn], constraints={...})
    VFS persistence:    Use /workspace/ for read/write operations
    LangGraph:          SandboxTool.from_functions([...]).as_langchain_tool()

    All recipes are self-contained - copy and adapt as needed!
    """)


if __name__ == "__main__":
    main()
