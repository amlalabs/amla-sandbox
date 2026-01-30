#!/usr/bin/env python3
"""Virtual Filesystem (VFS) Tutorial (VFS) Operations

This tutorial covers:
  1. Reading and writing files in the VFS
  2. Directory operations (mkdir, rmdir, listing)
  3. Persisting data between executions
  4. Using the VFS for agent state management
  5. Common VFS patterns for data processing

Prerequisites:
    uv pip install "git+https://github.com/amlalabs/amla-sandbox"

Run:
    python vfs_file_operations.py

Time: ~10 minutes

IMPORTANT: All fs operations are async and require `await`.
"""

from amla_sandbox import create_sandbox_tool

# =============================================================================
# Part 1: Basic File Operations
# =============================================================================


def part1_basic_io() -> None:
    """Read and write files in the virtual filesystem."""
    print("\n" + "=" * 60)
    print("Part 1: Basic File Operations")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # The VFS has two writable directories:
    #   /workspace/  - main working directory (ReadWrite)
    #   /tmp/        - temporary files (ReadWrite)
    #
    # Root (/) is READ-ONLY. Always create files under /workspace/ or /tmp/.
    # Common conventions:
    #   /workspace/data/   - working files and data
    #   /workspace/state/  - persistent state between agent turns
    #   /workspace/cache/  - cached computations

    # Write a text file (async - requires await)
    sandbox.run(
        """
        await fs.writeFile('/workspace/greeting.txt', 'Hello, VFS!');
        console.log("Wrote greeting.txt");
    """,
        language="javascript",
    )

    # Read it back
    output = sandbox.run(
        """
        const content = await fs.readFile('/workspace/greeting.txt');
        console.log(content);
    """,
        language="javascript",
    )
    print(f"Read file: {output}")

    # Write JSON data (very common pattern)
    sandbox.run(
        """
        const data = {
            users: [
                { id: 1, name: "Alice", role: "admin" },
                { id: 2, name: "Bob", role: "user" },
                { id: 3, name: "Charlie", role: "user" }
            ],
            lastUpdated: new Date().toISOString()
        };
        await fs.writeFile('/workspace/users.json', JSON.stringify(data, null, 2));
        console.log("Wrote users.json");
    """,
        language="javascript",
    )

    # Read and parse JSON
    output = sandbox.run(
        """
        const data = JSON.parse(await fs.readFile('/workspace/users.json'));
        const admins = data.users.filter(u => u.role === 'admin');
        console.log(`Found ${admins.length} admin(s): ${admins.map(a => a.name).join(', ')}`);
    """,
        language="javascript",
    )
    print(f"Parsed JSON: {output}")


# =============================================================================
# Part 2: Directory Operations
# =============================================================================


def part2_directories() -> None:
    """Work with directories in the VFS."""
    print("\n" + "=" * 60)
    print("Part 2: Directory Operations")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Create a directory structure (under /workspace which is writable)
    output = sandbox.run(
        """
        // Create nested directories (async operations)
        // Note: Always use /workspace/ or /tmp/ - root is read-only
        await fs.mkdir('/workspace/project', { recursive: true });
        await fs.mkdir('/workspace/project/src');
        await fs.mkdir('/workspace/project/data');
        await fs.mkdir('/workspace/project/output');

        // Create some files
        await fs.writeFile('/workspace/project/src/main.js', 'console.log("Hello");');
        await fs.writeFile('/workspace/project/src/utils.js', 'export function add(a, b) { return a + b; }');
        await fs.writeFile('/workspace/project/data/input.json', '{"items": [1, 2, 3]}');

        console.log("Created project structure");
    """,
        language="javascript",
    )
    print(output)

    # List directory contents
    output = sandbox.run(
        """
        const files = await fs.readdir('/workspace/project/src');
        console.log("Files in /workspace/project/src:");
        files.forEach(f => console.log("  - " + f));
    """,
        language="javascript",
    )
    print(output)

    # Check if file/directory exists
    output = sandbox.run(
        """
        async function exists(path) {
            try {
                await fs.stat(path);
                return true;
            } catch {
                return false;
            }
        }

        console.log(`/workspace/project exists: ${await exists('/workspace/project')}`);
        console.log(`/workspace/project/src exists: ${await exists('/workspace/project/src')}`);
        console.log(`/nonexistent exists: ${await exists('/nonexistent')}`);
    """,
        language="javascript",
    )
    print(output)

    # Get file stats
    output = sandbox.run(
        """
        const stats = await fs.stat('/workspace/project/src/main.js');
        console.log("File stats for main.js:");
        console.log(`  Size: ${stats.size} bytes`);
        console.log(`  Is file: ${stats.isFile()}`);
        console.log(`  Is directory: ${stats.isDirectory()}`);
    """,
        language="javascript",
    )
    print(output)


# =============================================================================
# Part 3: State Persistence Between Executions
# =============================================================================


def part3_persistence() -> None:
    """Use the VFS to maintain state between sandbox executions."""
    print("\n" + "=" * 60)
    print("Part 3: State Persistence")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Agents often need to maintain state between turns.
    # The /workspace/state/ directory is a convention for this.

    # Execution 1: Initialize state
    sandbox.run(
        """
        const state = {
            conversationId: "conv_" + Date.now(),
            turnCount: 0,
            memory: []
        };
        await fs.mkdir('/workspace/state', { recursive: true });
        await fs.writeFile('/workspace/state/agent.json', JSON.stringify(state));
        console.log("Initialized agent state");
    """,
        language="javascript",
    )
    print("Turn 1: Initialized state")

    # Execution 2: Update state
    output = sandbox.run(
        """
        // Load state
        const state = JSON.parse(await fs.readFile('/workspace/state/agent.json'));

        // Update
        state.turnCount++;
        state.memory.push({ turn: state.turnCount, action: "Searched for documents" });

        // Save
        await fs.writeFile('/workspace/state/agent.json', JSON.stringify(state));
        console.log(`Turn ${state.turnCount}: Updated state`);
    """,
        language="javascript",
    )
    print(f"Turn 2: {output}")

    # Execution 3: Another update
    output = sandbox.run(
        """
        const state = JSON.parse(await fs.readFile('/workspace/state/agent.json'));
        state.turnCount++;
        state.memory.push({ turn: state.turnCount, action: "Analyzed results" });
        await fs.writeFile('/workspace/state/agent.json', JSON.stringify(state));
        console.log(`Turn ${state.turnCount}: Updated state`);
    """,
        language="javascript",
    )
    print(f"Turn 3: {output}")

    # Execution 4: Read final state
    output = sandbox.run(
        """
        const state = JSON.parse(await fs.readFile('/workspace/state/agent.json'));
        console.log("Final state:");
        console.log(JSON.stringify(state, null, 2));
    """,
        language="javascript",
    )
    print(f"Final state:\n{output}")


# =============================================================================
# Part 4: Data Processing Patterns
# =============================================================================


def part4_data_processing() -> None:
    """Common patterns for processing data in the VFS."""
    print("\n" + "=" * 60)
    print("Part 4: Data Processing Patterns")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Pattern 1: Process large datasets in chunks
    print("\nPattern 1: Chunked Processing")
    output = sandbox.run(
        """
        // Simulate a large dataset
        const items = Array.from({length: 100}, (_, i) => ({
            id: i + 1,
            value: Math.random() * 100,
            category: ['A', 'B', 'C'][i % 3]
        }));

        await fs.writeFile('/workspace/large_dataset.json', JSON.stringify(items));

        // Process in chunks of 25
        const CHUNK_SIZE = 25;
        let processed = 0;
        const results = [];

        const data = JSON.parse(await fs.readFile('/workspace/large_dataset.json'));
        for (let i = 0; i < data.length; i += CHUNK_SIZE) {
            const chunk = data.slice(i, i + CHUNK_SIZE);
            const chunkResult = {
                chunk: Math.floor(i / CHUNK_SIZE) + 1,
                count: chunk.length,
                avgValue: chunk.reduce((sum, x) => sum + x.value, 0) / chunk.length
            };
            results.push(chunkResult);
            processed += chunk.length;
        }

        console.log(`Processed ${processed} items in ${results.length} chunks`);
        console.log(`Average values per chunk: ${results.map(r => r.avgValue.toFixed(1)).join(', ')}`);
    """,
        language="javascript",
    )
    print(output)

    # Pattern 2: Aggregate results from multiple files
    print("\nPattern 2: Multi-file Aggregation")
    output = sandbox.run(
        """
        // Create multiple result files (simulating distributed processing)
        await fs.mkdir('/workspace/results', { recursive: true });

        for (let region of ['us-east', 'us-west', 'europe']) {
            const regionData = {
                region: region,
                sales: Math.floor(Math.random() * 100000),
                customers: Math.floor(Math.random() * 1000)
            };
            await fs.writeFile(
                `/workspace/results/${region}.json`,
                JSON.stringify(regionData)
            );
        }

        // Aggregate all region files
        const files = await fs.readdir('/workspace/results');
        const aggregated = {
            totalSales: 0,
            totalCustomers: 0,
            byRegion: {}
        };

        for (const file of files) {
            const data = JSON.parse(
                await fs.readFile(`/workspace/results/${file}`)
            );
            aggregated.totalSales += data.sales;
            aggregated.totalCustomers += data.customers;
            aggregated.byRegion[data.region] = data;
        }

        console.log("Aggregated results:");
        console.log(`  Total Sales: $${aggregated.totalSales.toLocaleString()}`);
        console.log(`  Total Customers: ${aggregated.totalCustomers}`);
    """,
        language="javascript",
    )
    print(output)

    # Pattern 3: Cache expensive computations
    print("\nPattern 3: Computation Caching")
    sandbox.run(
        """
        await fs.mkdir('/workspace/cache', { recursive: true });

        async function cachedCompute(key, computeFn) {
            const cachePath = `/workspace/cache/${key}.json`;
            try {
                // Try to read from cache
                const cached = JSON.parse(await fs.readFile(cachePath));
                console.log(`Cache HIT for ${key}`);
                return cached;
            } catch {
                // Cache miss - compute and store
                console.log(`Cache MISS for ${key} - computing...`);
                const result = computeFn();
                await fs.writeFile(cachePath, JSON.stringify(result));
                return result;
            }
        }

        // First call - cache miss
        const result1 = await cachedCompute('fibonacci_10', () => {
            function fib(n) {
                return n <= 1 ? n : fib(n-1) + fib(n-2);
            }
            return { n: 10, value: fib(10) };
        });

        // Second call - cache hit
        const result2 = await cachedCompute('fibonacci_10', () => {
            // This won't run - cached!
            throw new Error("Should not compute again!");
        });

        console.log(`Result: fib(10) = ${result1.value}`);
    """,
        language="javascript",
    )


# =============================================================================
# Part 5: Binary Data and Encoding
# =============================================================================


def part5_binary_data() -> None:
    """Work with binary data in the VFS."""
    print("\n" + "=" * 60)
    print("Part 5: Binary Data and Encoding")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Base64 encoding/decoding
    output = sandbox.run(
        """
        // Encode text to base64
        const text = "Hello, this is some text to encode!";
        const encoded = btoa(text);
        console.log(`Original: ${text}`);
        console.log(`Base64: ${encoded}`);

        // Decode back
        const decoded = atob(encoded);
        console.log(`Decoded: ${decoded}`);
        console.log(`Match: ${text === decoded}`);
    """,
        language="javascript",
    )
    print(output)

    # Store and retrieve binary-like data
    output = sandbox.run(
        """
        // Create some "binary" data (array of bytes)
        const binaryData = new Uint8Array([0x48, 0x65, 0x6c, 0x6c, 0x6f]);  // "Hello"

        // Convert to base64 for storage
        const base64 = btoa(String.fromCharCode(...binaryData));
        await fs.writeFile('/workspace/binary.b64', base64);

        // Read and decode
        const storedBase64 = await fs.readFile('/workspace/binary.b64');
        const decodedStr = atob(storedBase64);
        const decodedBytes = new Uint8Array([...decodedStr].map(c => c.charCodeAt(0)));

        console.log(`Stored bytes: [${binaryData.join(', ')}]`);
        console.log(`Decoded bytes: [${decodedBytes.join(', ')}]`);
        console.log(`As string: "${decodedStr}"`);
    """,
        language="javascript",
    )
    print(output)


# =============================================================================
# Part 6: Real-World Example - Agent Memory
# =============================================================================


def part6_agent_memory() -> None:
    """Build a simple agent memory system using the VFS."""
    print("\n" + "=" * 60)
    print("Part 6: Real-World Example - Agent Memory")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Initialize the memory system
    sandbox.run(
        """
        // Agent Memory System
        // Stores conversation history, facts, and working memory

        await fs.mkdir('/workspace/memory', { recursive: true });
        await fs.mkdir('/workspace/memory/facts', { recursive: true });
        await fs.mkdir('/workspace/memory/working', { recursive: true });

        // Initialize conversation history
        await fs.writeFile('/workspace/memory/history.json', JSON.stringify({
            messages: [],
            startedAt: new Date().toISOString()
        }));

        console.log("Memory system initialized");
    """,
        language="javascript",
    )

    # Simulate a few conversation turns
    print("\nSimulating conversation turns...")

    # Turn 1: User asks a question
    sandbox.run(
        """
        const history = JSON.parse(await fs.readFile('/workspace/memory/history.json'));
        history.messages.push({
            role: 'user',
            content: 'What is the capital of France?',
            timestamp: new Date().toISOString()
        });
        history.messages.push({
            role: 'assistant',
            content: 'The capital of France is Paris.',
            timestamp: new Date().toISOString()
        });

        // Store a fact we learned
        await fs.writeFile('/workspace/memory/facts/france_capital.json', JSON.stringify({
            subject: 'France',
            predicate: 'capital',
            object: 'Paris',
            source: 'conversation',
            confidence: 1.0
        }));

        await fs.writeFile('/workspace/memory/history.json', JSON.stringify(history));
        console.log("Turn 1 complete");
    """,
        language="javascript",
    )
    print("  Turn 1: Asked about France's capital")

    # Turn 2: Follow-up question
    sandbox.run(
        """
        const history = JSON.parse(await fs.readFile('/workspace/memory/history.json'));
        history.messages.push({
            role: 'user',
            content: 'What about Germany?',
            timestamp: new Date().toISOString()
        });
        history.messages.push({
            role: 'assistant',
            content: 'The capital of Germany is Berlin.',
            timestamp: new Date().toISOString()
        });

        await fs.writeFile('/workspace/memory/facts/germany_capital.json', JSON.stringify({
            subject: 'Germany',
            predicate: 'capital',
            object: 'Berlin',
            source: 'conversation',
            confidence: 1.0
        }));

        await fs.writeFile('/workspace/memory/history.json', JSON.stringify(history));
        console.log("Turn 2 complete");
    """,
        language="javascript",
    )
    print("  Turn 2: Asked about Germany's capital")

    # Query the memory
    output = sandbox.run(
        """
        // Query conversation history
        const history = JSON.parse(await fs.readFile('/workspace/memory/history.json'));
        console.log(`Conversation has ${history.messages.length} messages`);
        console.log(`Started at: ${history.startedAt}`);

        // Query stored facts
        const factFiles = await fs.readdir('/workspace/memory/facts');
        console.log(`\\nStored facts (${factFiles.length}):`);
        for (const file of factFiles) {
            const fact = JSON.parse(await fs.readFile(`/workspace/memory/facts/${file}`));
            console.log(`  - ${fact.subject}'s ${fact.predicate} is ${fact.object}`);
        }

        // Show last exchange
        const lastUser = history.messages.filter(m => m.role === 'user').slice(-1)[0];
        const lastAssistant = history.messages.filter(m => m.role === 'assistant').slice(-1)[0];
        console.log(`\\nLast exchange:`);
        console.log(`  User: "${lastUser.content}"`);
        console.log(`  Assistant: "${lastAssistant.content}"`);
    """,
        language="javascript",
    )
    print(f"\nMemory query results:\n{output}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all tutorial parts."""
    print("=" * 60)
    print("Virtual Filesystem (VFS) Tutorial (VFS) Operations")
    print("=" * 60)

    part1_basic_io()
    part2_directories()
    part3_persistence()
    part4_data_processing()
    part5_binary_data()
    part6_agent_memory()

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)
    print("""
Key takeaways:

1. The VFS is an in-memory filesystem inside the sandbox
2. All fs operations are ASYNC - use `await fs.readFile()` and `await fs.writeFile()`
3. JSON is the natural format for structured data
4. State persists between run() calls on the SAME sandbox instance
5. Root (/) is READ-ONLY - use /workspace/ or /tmp/ for all writes
6. Common conventions: /workspace/data/, /workspace/state/, /workspace/cache/

VFS is the foundation for:
- Agent state management
- Multi-step data processing
- Caching expensive computations
- Building memory systems

Next tutorial: 03_shell_pipelines.py
    Master shell utilities for data processing.
    """)


if __name__ == "__main__":
    main()
