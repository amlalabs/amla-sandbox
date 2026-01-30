#!/usr/bin/env python3
"""Shell Pipelines Tutorial - Data Processing with Unix Utilities

This tutorial covers:
  1. Available shell utilities in the sandbox
  2. Text search with grep
  3. Character translation with tr
  4. Field extraction with cut
  5. JSON processing with jq (the most powerful tool!)
  6. Data aggregation (sort, uniq, wc)
  7. Complete pipeline examples
  8. Combining shell and JavaScript

Prerequisites:
    uv pip install "git+https://github.com/amlalabs/amla-sandbox"

Run:
    python shell_pipelines.py

Time: ~15 minutes
"""

from amla_sandbox import create_sandbox_tool

# =============================================================================
# Part 1: Available Shell Utilities
# =============================================================================


def part1_available_utilities() -> None:
    """Overview of shell utilities available in the sandbox."""
    print("\n" + "=" * 60)
    print("Part 1: Available Shell Utilities")
    print("=" * 60)

    print("""
The amla-sandbox includes these built-in utilities:

TEXT PROCESSING:
  grep    - Search for patterns in text (supports -E, -i, -v, -c)
  tr      - Translate or delete characters
  cut     - Extract fields from lines

DATA UTILITIES:
  sort    - Sort lines of text (-n, -r, -k, -t)
  uniq    - Filter adjacent duplicate lines (-c)
  head    - Output first N lines (-n)
  tail    - Output last N lines (-n, +n)
  wc      - Word, line, character count (-l, -w, -c)

JSON PROCESSING:
  jq      - JSON processor (very powerful!)

FILE UTILITIES:
  cat     - Concatenate and print files
  echo    - Display a line of text

Note: These run INSIDE the WASM sandbox, not as subprocess calls.
There's no shell escape possible.
    """)

    sandbox = create_sandbox_tool()

    # Quick demo of each - use language="shell" for shell commands
    print("Quick demonstrations:")

    output = sandbox.run("echo 'hello world' | grep 'world'", language="shell")
    print(f"  grep: {output.strip()}")

    output = sandbox.run("echo 'hello' | tr 'a-z' 'A-Z'", language="shell")
    print(f"  tr: {output.strip()}")

    output = sandbox.run("echo 'apple,banana,cherry' | cut -d',' -f2", language="shell")
    print(f"  cut: {output.strip()}")

    output = sandbox.run("echo -e 'c\\na\\nb' | sort", language="shell")
    print(f"  sort: {output.strip().replace(chr(10), ', ')}")

    output = sandbox.run("echo 'hello world' | wc -w", language="shell")
    print(f"  wc -w: {output.strip()} words")

    output = sandbox.run("echo '[1,2,3]' | jq '.[1]'", language="shell")
    print(f"  jq: {output.strip()}")


# =============================================================================
# Part 2: Text Search with grep
# =============================================================================


def part2_grep() -> None:
    """Search and filter text with grep."""
    print("\n" + "=" * 60)
    print("Part 2: Text Search with grep")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Create sample data using JavaScript
    sandbox.run(
        """
        const logData = `2024-01-15 10:00:01 INFO  Server started on port 8080
2024-01-15 10:00:05 DEBUG Loading configuration
2024-01-15 10:00:10 INFO  Connected to database
2024-01-15 10:01:00 WARN  High memory usage detected
2024-01-15 10:01:15 ERROR Connection timeout to service X
2024-01-15 10:01:30 INFO  Retrying connection...
2024-01-15 10:01:45 INFO  Connection restored
2024-01-15 10:02:00 ERROR Failed to process request: invalid input
2024-01-15 10:02:30 DEBUG Request completed in 150ms
2024-01-15 10:03:00 INFO  Shutting down gracefully`;
        await fs.writeFile('/workspace/server.log', logData);
        console.log("Created server.log");
    """,
        language="javascript",
    )

    # Basic grep - find all errors
    print("\nFind all ERROR lines:")
    output = sandbox.run("cat /workspace/server.log | grep 'ERROR'", language="shell")
    print(f"  {output.strip().replace(chr(10), chr(10) + '  ')}")

    # Case-insensitive search (-i)
    print("\nCase-insensitive search for 'error' or 'warn':")
    output = sandbox.run(
        "cat /workspace/server.log | grep -iE 'error|warn'", language="shell"
    )
    print(f"  {output.strip().replace(chr(10), chr(10) + '  ')}")

    # Invert match (-v) - show non-DEBUG lines
    print("\nAll non-DEBUG lines (first 3):")
    output = sandbox.run(
        "cat /workspace/server.log | grep -v 'DEBUG' | head -3", language="shell"
    )
    for line in output.strip().split("\n"):
        print(f"  {line}")

    # Count matches (-c)
    print("\nCount of each log level:")
    output = sandbox.run("cat /workspace/server.log | grep -c 'INFO'", language="shell")
    print(f"  INFO:  {output.strip()}")
    output = sandbox.run(
        "cat /workspace/server.log | grep -c 'DEBUG'", language="shell"
    )
    print(f"  DEBUG: {output.strip()}")
    output = sandbox.run(
        "cat /workspace/server.log | grep -c 'ERROR'", language="shell"
    )
    print(f"  ERROR: {output.strip()}")


# =============================================================================
# Part 3: Character Translation with tr
# =============================================================================


def part3_tr() -> None:
    """Transform text with tr."""
    print("\n" + "=" * 60)
    print("Part 3: Character Translation with tr")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    print("tr examples:")

    # Uppercase
    output = sandbox.run("echo 'hello world' | tr 'a-z' 'A-Z'", language="shell")
    print(f"  Uppercase: {output.strip()}")

    # Lowercase
    output = sandbox.run("echo 'HELLO WORLD' | tr 'A-Z' 'a-z'", language="shell")
    print(f"  Lowercase: {output.strip()}")

    # Delete characters (-d)
    output = sandbox.run("echo 'hello 123 world 456' | tr -d '0-9'", language="shell")
    print(f"  Delete digits: {output.strip()}")

    # Replace characters
    output = sandbox.run("echo 'hello-world_test' | tr '-_' '  '", language="shell")
    print(f"  Replace chars: {output.strip()}")

    # Practical: Normalize whitespace
    print("\nPractical: Normalize CSV separators")
    output = sandbox.run("echo 'a;b;c' | tr ';' ','", language="shell")
    print(f"  Normalized: {output.strip()}")


# =============================================================================
# Part 4: Field Extraction with cut
# =============================================================================


def part4_cut() -> None:
    """Extract fields from structured data with cut."""
    print("\n" + "=" * 60)
    print("Part 4: Field Extraction with cut")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Create sample CSV data
    sandbox.run(
        """
        const csv = `name,department,salary
Alice,Engineering,95000
Bob,Sales,75000
Charlie,Engineering,85000
Diana,Marketing,70000`;
        await fs.writeFile('/workspace/employees.csv', csv);
        console.log("Created employees.csv");
    """,
        language="javascript",
    )

    print("\ncut examples:")

    # Extract column by position
    output = sandbox.run(
        "cat /workspace/employees.csv | cut -d',' -f1 | head -4", language="shell"
    )
    print(f"  Column 1 (name): {output.strip().replace(chr(10), ', ')}")

    # Extract column 2
    output = sandbox.run(
        "cat /workspace/employees.csv | cut -d',' -f2 | head -4", language="shell"
    )
    print(f"  Column 2 (dept): {output.strip().replace(chr(10), ', ')}")

    # Multiple columns
    output = sandbox.run(
        "cat /workspace/employees.csv | cut -d',' -f1,3 | head -4", language="shell"
    )
    print(f"  Columns 1,3: {output.strip().replace(chr(10), ' | ')}")

    # Character positions (-c)
    output = sandbox.run("echo 'Hello World' | cut -c1-5", language="shell")
    print(f"  First 5 chars: {output.strip()}")


# =============================================================================
# Part 5: JSON Processing with jq
# =============================================================================


def part5_jq() -> None:
    """Process JSON data with jq - the most powerful tool!"""
    print("\n" + "=" * 60)
    print("Part 5: JSON Processing with jq")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Create sample JSON data
    sandbox.run(
        """
        const data = {
            "users": [
                {"id": 1, "name": "Alice", "role": "admin", "active": true, "salary": 95000},
                {"id": 2, "name": "Bob", "role": "user", "active": true, "salary": 75000},
                {"id": 3, "name": "Charlie", "role": "user", "active": false, "salary": 85000},
                {"id": 4, "name": "Diana", "role": "moderator", "active": true, "salary": 70000}
            ],
            "metadata": {"version": "1.0", "generated": "2024-01-15"}
        };
        await fs.writeFile('/workspace/users.json', JSON.stringify(data));
        console.log("Created users.json");
    """,
        language="javascript",
    )

    print("\njq examples:")

    # Access nested fields
    output = sandbox.run("cat /workspace/users.json | jq '.metadata'", language="shell")
    print(f"  Access nested: {output.strip()}")

    # Array access
    output = sandbox.run(
        "cat /workspace/users.json | jq '.users[0].name'", language="shell"
    )
    print(f"  First user: {output.strip()}")

    # Get all names
    output = sandbox.run(
        "cat /workspace/users.json | jq '.users[].name'", language="shell"
    )
    print(f"  All names: {output.strip().replace(chr(10), ', ')}")

    # Filter array (like WHERE clause)
    output = sandbox.run(
        "cat /workspace/users.json | jq '.users[] | select(.active == true) | .name'",
        language="shell",
    )
    print(f"  Active users: {output.strip().replace(chr(10), ', ')}")

    # Count
    output = sandbox.run(
        "cat /workspace/users.json | jq '.users | length'", language="shell"
    )
    print(f"  User count: {output.strip()}")

    # Sum (like SQL SUM)
    output = sandbox.run(
        "cat /workspace/users.json | jq '[.users[].salary] | add'", language="shell"
    )
    print(f"  Total salary: ${output.strip()}")

    # Sort
    output = sandbox.run(
        "cat /workspace/users.json | jq '.users | sort_by(.salary) | reverse | .[].name'",
        language="shell",
    )
    print(f"  By salary (desc): {output.strip().replace(chr(10), ', ')}")


# =============================================================================
# Part 6: Sorting and Deduplication
# =============================================================================


def part6_sort_and_uniq() -> None:
    """Sort data and remove duplicates."""
    print("\n" + "=" * 60)
    print("Part 6: Sorting and Deduplication")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Create sample data
    sandbox.run(
        """
        await fs.writeFile('/workspace/access.log', `192.168.1.100
192.168.1.105
192.168.1.100
10.0.0.50
192.168.1.105
192.168.1.100
10.0.0.50
192.168.1.200
192.168.1.100`);
        console.log("Created access.log");
    """,
        language="javascript",
    )

    print("\nsort examples:")

    # Basic sort
    output = sandbox.run("echo -e 'banana\\napple\\ncherry' | sort", language="shell")
    print(f"  Alphabetical: {output.strip().replace(chr(10), ', ')}")

    # Reverse sort
    output = sandbox.run(
        "echo -e 'banana\\napple\\ncherry' | sort -r", language="shell"
    )
    print(f"  Reverse: {output.strip().replace(chr(10), ', ')}")

    # Numeric sort
    output = sandbox.run("echo -e '10\\n2\\n100\\n5' | sort -n", language="shell")
    print(f"  Numeric: {output.strip().replace(chr(10), ', ')}")

    print("\nuniq examples:")

    # Remove adjacent duplicates (requires sorted input)
    output = sandbox.run("cat /workspace/access.log | sort | uniq", language="shell")
    print(f"  Unique IPs: {output.strip().replace(chr(10), ', ')}")

    # Count occurrences
    output = sandbox.run(
        "cat /workspace/access.log | sort | uniq -c | sort -rn", language="shell"
    )
    print(f"  IP frequency:\n  {output.strip().replace(chr(10), chr(10) + '  ')}")

    # Top N pattern (very common!)
    print("\nTop 2 most frequent IPs:")
    output = sandbox.run(
        "cat /workspace/access.log | sort | uniq -c | sort -rn | head -2",
        language="shell",
    )
    print(f"  {output.strip().replace(chr(10), chr(10) + '  ')}")


# =============================================================================
# Part 7: Complete Pipeline Example
# =============================================================================


def part7_complete_pipeline() -> None:
    """Build a complete data processing pipeline using jq."""
    print("\n" + "=" * 60)
    print("Part 7: Complete Pipeline Example")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Create realistic log data as JSON
    sandbox.run(
        """
        const logs = [];
        const endpoints = ['/api/users', '/api/orders', '/api/products', '/api/auth'];
        const statuses = [200, 200, 200, 200, 201, 400, 401, 404, 500];

        for (let i = 0; i < 50; i++) {
            const endpoint = endpoints[Math.floor(Math.random() * endpoints.length)];
            const status = statuses[Math.floor(Math.random() * statuses.length)];
            const duration = Math.floor(Math.random() * 500) + 10;
            logs.push({endpoint, status, duration_ms: duration});
        }

        await fs.writeFile('/workspace/api.log', logs.map(l => JSON.stringify(l)).join('\\n'));
        console.log("Created api.log with " + logs.length + " entries");
    """,
        language="javascript",
    )

    print("\nAnalyzing API logs with jq...")

    # Analysis 1: Request distribution by endpoint
    print("\n1. Requests by endpoint:")
    output = sandbox.run(
        "cat /workspace/api.log | jq -s 'group_by(.endpoint) | map({endpoint: .[0].endpoint, count: length}) | sort_by(.count) | reverse'",
        language="shell",
    )
    print(f"  {output.strip()}")

    # Analysis 2: Error rate (4xx and 5xx)
    print("\n2. Error analysis:")
    output = sandbox.run("cat /workspace/api.log | jq -s 'length'", language="shell")
    total = output.strip()
    output = sandbox.run(
        "cat /workspace/api.log | jq -s '[.[] | select(.status >= 400)] | length'",
        language="shell",
    )
    errors = output.strip()
    print(f"  Total requests: {total}")
    print(f"  Errors (4xx/5xx): {errors}")

    # Analysis 3: Average response time by endpoint
    print("\n3. Average response time by endpoint:")
    output = sandbox.run(
        """cat /workspace/api.log | jq -s 'group_by(.endpoint) | map({endpoint: .[0].endpoint, avg_ms: ([.[].duration_ms] | add / length | floor)}) | sort_by(.avg_ms)'""",
        language="shell",
    )
    print(f"  {output.strip()}")


# =============================================================================
# Part 8: Combining Shell and JavaScript
# =============================================================================


def part8_shell_and_js() -> None:
    """Combine shell pipelines with JavaScript for complex logic."""
    print("\n" + "=" * 60)
    print("Part 8: Combining Shell and JavaScript")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Use shell for text processing, JS for complex logic
    output = sandbox.run(
        """
        // Create sample order data
        const orders = [
            {id: 1, customer: "Alice", items: [{name: "Widget", price: 25.99, qty: 2}], status: "completed"},
            {id: 2, customer: "Bob", items: [{name: "Gadget", price: 49.99, qty: 1}, {name: "Widget", price: 25.99, qty: 3}], status: "pending"},
            {id: 3, customer: "Charlie", items: [{name: "Gadget", price: 49.99, qty: 2}], status: "completed"},
            {id: 4, customer: "Alice", items: [{name: "Thing", price: 15.00, qty: 5}], status: "completed"},
        ];

        // Write as NDJSON for shell processing
        await fs.writeFile('/workspace/orders.ndjson',
            orders.map(o => JSON.stringify(o)).join('\\n')
        );

        // Use shell() to run shell commands from JavaScript
        const completedJson = shell('cat /workspace/orders.ndjson | jq -c "select(.status == \\\\"completed\\\\")"').trim();
        const completedOrders = completedJson.split('\\n').filter(l => l).map(line => JSON.parse(line));

        // Use JavaScript for complex calculations
        const customerTotals = {};
        for (const order of completedOrders) {
            const customer = order.customer;
            const orderTotal = order.items.reduce((sum, item) => sum + item.price * item.qty, 0);

            if (!customerTotals[customer]) {
                customerTotals[customer] = { orders: 0, total: 0 };
            }
            customerTotals[customer].orders++;
            customerTotals[customer].total += orderTotal;
        }

        console.log("Customer totals (completed orders only):");
        for (const [customer, data] of Object.entries(customerTotals)) {
            console.log("  " + customer + ": " + data.orders + " orders, $" + data.total.toFixed(2));
        }
    """,
        language="javascript",
    )
    print(output)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all tutorial parts."""
    print("=" * 60)
    print("Shell Pipelines Tutorial")
    print("Data Processing with Unix Utilities")
    print("=" * 60)

    part1_available_utilities()
    part2_grep()
    part3_tr()
    part4_cut()
    part5_jq()
    part6_sort_and_uniq()
    part7_complete_pipeline()
    part8_shell_and_js()

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)
    print("""
Key takeaways:

1. Use language="shell" for shell commands
2. grep - search and filter text (supports regex with -E)
3. tr - character translation and deletion
4. cut - extract fields by delimiter
5. jq - JSON processing (most powerful tool!)
6. sort/uniq - aggregation and deduplication
7. Combine shell and JS with shell() function inside JavaScript

Common patterns:
  - Log analysis with grep
  - JSON transformation with jq
  - Top-N queries with sort/uniq/head

Pro tip: jq can do almost anything! Use it for:
  - Filtering: select(.field == value)
  - Mapping: map({new_field: .old_field})
  - Aggregating: group_by(.field) | map(...)
  - Sorting: sort_by(.field)
  - Math: [.items[].price] | add
    """)


if __name__ == "__main__":
    main()
