#!/usr/bin/env python3
"""Error Handling Tutorial and Capability Violations

This tutorial covers:
  1. Types of errors in amla-sandbox
  2. Capability violation handling
  3. JavaScript error handling patterns
  4. Tool handler error strategies
  5. Graceful degradation patterns
  6. Debugging and troubleshooting

Prerequisites:
    uv pip install "git+https://github.com/amlalabs/amla-sandbox"

Run:
    python error_handling.py

Time: ~15 minutes
"""

from typing import Any

from amla_sandbox import (
    Sandbox,
    MethodCapability,
    ConstraintSet,
    Param,
    ToolDefinition,
    CapabilityError,
    create_sandbox_tool,
)

# =============================================================================
# Part 1: Types of Errors
# =============================================================================


def part1_error_types() -> None:
    """Overview of error types in amla-sandbox."""
    print("\n" + "=" * 60)
    print("Part 1: Types of Errors")
    print("=" * 60)

    print("""
amla-sandbox has several categories of errors:

1. CAPABILITY ERRORS (Python-side)
   - CapabilityError: Base class for all capability violations
   - CallLimitExceededError: max_calls budget exhausted
   - ConstraintError: Parameter constraint violated
   - MethodNotAllowedError: No capability for the method

2. JAVASCRIPT ERRORS (Sandbox-side)
   - Syntax errors in JS code
   - Runtime errors (TypeError, ReferenceError, etc.)
   - Thrown errors (throw new Error(...))

3. TOOL HANDLER ERRORS (Python-side)
   - Exceptions raised by your tool implementations
   - These propagate to JavaScript as errors

4. RUNTIME ERRORS
   - Sandbox initialization failures
   - Resource exhaustion
    """)


# =============================================================================
# Part 2: Capability Violations
# =============================================================================


def part2_capability_violations() -> None:
    """Handle capability-related errors."""
    print("\n" + "=" * 60)
    print("Part 2: Capability Violations")
    print("=" * 60)

    tools = [
        ToolDefinition(
            name="transfer",
            description="Transfer funds",
            parameters={
                "type": "object",
                "properties": {"amount": {"type": "number"}, "to": {"type": "string"}},
            },
        )
    ]

    def handler(_method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"transferred": params["amount"], "to": params["to"]}

    # Sandbox with limited capabilities
    sandbox = Sandbox(
        tools=tools,
        capabilities=[
            MethodCapability(
                method_pattern="transfer",
                constraints=ConstraintSet(
                    [
                        Param("amount") <= 1000,
                        Param("amount") > 0,
                    ]
                ),
                max_calls=3,
            )
        ],
        tool_handler=handler,
    )

    # Demo 1: Constraint violation
    print("\n1. Constraint Violation (amount > 1000):")
    try:
        # Using can_call first is the recommended pattern
        if not sandbox.can_call("transfer", {"amount": 5000, "to": "alice"}):
            print("   Pre-check: Call would be denied")

        # If you try anyway, you get an error in JS
        result = sandbox.execute("""
            try {
                const r = await transfer({amount: 5000, to: "alice"});
                console.log("Unexpected success:", r);
            } catch (e) {
                console.log("Caught in JS:", e.message);
            }
        """)
        print(f"   {result.strip()}")
    except CapabilityError as e:
        print(f"   Caught in Python: {e}")

    # Demo 2: Negative amount
    print("\n2. Constraint Violation (amount <= 0):")
    result = sandbox.execute("""
        try {
            const r = await transfer({amount: -100, to: "bob"});
            console.log("Unexpected success:", r);
        } catch (e) {
            console.log("Caught in JS:", e.message);
        }
    """)
    print(f"   {result.strip()}")

    # Demo 3: Budget exhaustion
    print("\n3. Call Budget Exhaustion:")
    # Make 3 valid calls to exhaust budget
    for i in range(3):
        sandbox.execute(f"""
            await transfer({{amount: 100, to: "user{i}"}});
        """)
        print(
            f"   Call {i + 1}: Success (remaining: {sandbox.get_remaining_calls('cap:method:transfer')})"
        )

    # Fourth call should fail
    result = sandbox.execute("""
        try {
            const r = await transfer({amount: 100, to: "user4"});
            console.log("Unexpected success");
        } catch (e) {
            console.log("Caught in JS:", e.message);
        }
    """)
    print(f"   Call 4: {result.strip()}")


# =============================================================================
# Part 3: JavaScript Error Handling
# =============================================================================


def part3_js_errors() -> None:
    """Handle JavaScript errors."""
    print("\n" + "=" * 60)
    print("Part 3: JavaScript Error Handling")
    print("=" * 60)

    sandbox = create_sandbox_tool()

    # Syntax errors
    print("\n1. Syntax Errors:")
    try:
        sandbox.run("const x = {", language="javascript")  # Invalid syntax
    except Exception as e:
        print(f"   Caught: {type(e).__name__}")

    # Runtime errors
    print("\n2. Runtime Errors:")
    result = sandbox.run(
        """
        try {
            const obj = null;
            obj.property;  // TypeError: null has no properties
        } catch (e) {
            console.log("Caught:", e.name, "-", e.message);
        }
    """,
        language="javascript",
    )
    print(f"   {result.strip()}")

    # Reference errors
    print("\n3. Reference Errors:")
    result = sandbox.run(
        """
        try {
            undefinedVariable;  // ReferenceError
        } catch (e) {
            console.log("Caught:", e.name, "-", e.message);
        }
    """,
        language="javascript",
    )
    print(f"   {result.strip()}")

    # Custom thrown errors
    print("\n4. Custom Thrown Errors:")
    result = sandbox.run(
        """
        function validateAge(age) {
            if (age < 0) throw new Error("Age cannot be negative");
            if (age > 150) throw new Error("Age seems unrealistic");
            return true;
        }

        try {
            validateAge(-5);
        } catch (e) {
            console.log("Validation failed:", e.message);
        }
    """,
        language="javascript",
    )
    print(f"   {result.strip()}")


# =============================================================================
# Part 4: Tool Handler Errors
# =============================================================================


def part4_tool_handler_errors() -> None:
    """Handle errors from tool implementations."""
    print("\n" + "=" * 60)
    print("Part 4: Tool Handler Errors")
    print("=" * 60)

    class NetworkError(Exception):
        """Custom network error."""

        pass

    class ValidationError(Exception):
        """Custom validation error."""

        pass

    def fetch_data(source: str = "") -> dict[str, Any]:
        """Fetch data from a source."""
        if source == "unreachable":
            raise NetworkError(f"Cannot connect to {source}")
        if source == "timeout":
            raise TimeoutError("Request timed out after 30s")
        return {"data": f"Data from {source}"}

    def validate(value: Any = None) -> dict[str, Any]:
        """Validate a value."""
        if value is None:
            raise ValidationError("Value is required")
        if not isinstance(value, int):
            raise ValidationError(f"Expected int, got {type(value).__name__}")
        return {"valid": True, "value": value}

    def divide(a: float = 0, b: float = 0) -> dict[str, Any]:
        """Divide two numbers."""
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return {"result": a / b}

    # Use create_sandbox_tool for proper PCA handling
    sandbox = create_sandbox_tool(tools=[fetch_data, validate, divide])
    sandbox = sandbox.sandbox

    # Network error
    print("\n1. Network Error:")
    result = sandbox.execute("""
        try {
            const data = await fetch_data({source: "unreachable"});
            console.log("Got data:", data);
        } catch (e) {
            console.log("Error:", e.message);
        }
    """)
    print(f"   {result.strip()}")

    # Timeout error
    print("\n2. Timeout Error:")
    result = sandbox.execute("""
        try {
            const data = await fetch_data({source: "timeout"});
            console.log("Got data:", data);
        } catch (e) {
            console.log("Error:", e.message);
        }
    """)
    print(f"   {result.strip()}")

    # Validation error
    print("\n3. Validation Error:")
    result = sandbox.execute("""
        try {
            const result = await validate({value: "not an int"});
            console.log("Valid:", result);
        } catch (e) {
            console.log("Error:", e.message);
        }
    """)
    print(f"   {result.strip()}")

    # Division by zero
    print("\n4. Division by Zero:")
    result = sandbox.execute("""
        try {
            const result = await divide({a: 10, b: 0});
            console.log("Result:", result);
        } catch (e) {
            console.log("Error:", e.message);
        }
    """)
    print(f"   {result.strip()}")


# =============================================================================
# Part 5: Graceful Degradation Patterns
# =============================================================================


def part5_graceful_degradation() -> None:
    """Patterns for graceful error handling."""
    print("\n" + "=" * 60)
    print("Part 5: Graceful Degradation Patterns")
    print("=" * 60)

    # Pattern 1: Retry with exponential backoff
    print("\n1. Retry Pattern:")
    sandbox = create_sandbox_tool()
    result = sandbox.run(
        """
        async function fetchWithRetry(fn, maxRetries = 3) {
            let lastError;
            for (let attempt = 1; attempt <= maxRetries; attempt++) {
                try {
                    return await fn();
                } catch (e) {
                    lastError = e;
                    console.log(`Attempt ${attempt} failed: ${e.message}`);
                    // In real code, you'd add delay here
                }
            }
            throw lastError;
        }

        let callCount = 0;
        async function unreliableOperation() {
            callCount++;
            if (callCount < 3) {
                throw new Error("Service temporarily unavailable");
            }
            return {success: true, message: "Finally worked!"};
        }

        try {
            const result = await fetchWithRetry(unreliableOperation);
            console.log("Success:", result.message);
        } catch (e) {
            console.log("All retries failed:", e.message);
        }
    """,
        language="javascript",
    )
    print(f"   {result}")

    # Pattern 2: Fallback values
    print("\n2. Fallback Pattern:")
    result = sandbox.run(
        """
        async function getDataWithFallback(primary, fallback) {
            try {
                return await primary();
            } catch (e) {
                console.log("Primary failed, using fallback");
                return fallback;
            }
        }

        const data = await getDataWithFallback(
            async () => { throw new Error("Primary source down"); },
            {source: "cached", data: [1, 2, 3]}
        );
        console.log("Got data:", JSON.stringify(data));
    """,
        language="javascript",
    )
    print(f"   {result}")

    # Pattern 3: Circuit breaker
    print("\n3. Circuit Breaker Pattern:")
    result = sandbox.run(
        """
        class CircuitBreaker {
            constructor(threshold = 3, resetTimeout = 5000) {
                this.failures = 0;
                this.threshold = threshold;
                this.state = 'CLOSED';
                this.resetTimeout = resetTimeout;
            }

            async call(fn) {
                if (this.state === 'OPEN') {
                    throw new Error('Circuit breaker is OPEN');
                }

                try {
                    const result = await fn();
                    this.failures = 0;
                    return result;
                } catch (e) {
                    this.failures++;
                    if (this.failures >= this.threshold) {
                        this.state = 'OPEN';
                        console.log('Circuit breaker tripped!');
                    }
                    throw e;
                }
            }
        }

        const breaker = new CircuitBreaker(2);
        const failingFn = async () => { throw new Error("Service error"); };

        for (let i = 1; i <= 4; i++) {
            try {
                await breaker.call(failingFn);
            } catch (e) {
                console.log(`Call ${i}: ${e.message} (state: ${breaker.state})`);
            }
        }
    """,
        language="javascript",
    )
    print(f"   {result}")

    # Pattern 4: Result type pattern
    print("\n4. Result Type Pattern:")
    result = sandbox.run(
        """
        // Return {ok, value/error} instead of throwing
        async function safeOperation(fn) {
            try {
                const value = await fn();
                return {ok: true, value};
            } catch (e) {
                return {ok: false, error: e.message};
            }
        }

        const result1 = await safeOperation(async () => ({data: "success"}));
        console.log("Result 1:", JSON.stringify(result1));

        const result2 = await safeOperation(async () => { throw new Error("Failed"); });
        console.log("Result 2:", JSON.stringify(result2));

        // Easy to handle
        if (result2.ok) {
            console.log("Using value:", result2.value);
        } else {
            console.log("Handling error:", result2.error);
        }
    """,
        language="javascript",
    )
    print(f"   {result}")


# =============================================================================
# Part 6: Pre-flight Checks
# =============================================================================


def part6_preflight_checks() -> None:
    """Use can_call() for pre-flight authorization checks."""
    print("\n" + "=" * 60)
    print("Part 6: Pre-flight Checks")
    print("=" * 60)

    def payment_send(amount: float) -> dict[str, Any]:
        """Send a payment."""
        return {"sent": amount}

    # Use create_sandbox_tool with constraints
    sandbox = create_sandbox_tool(
        tools=[payment_send],
        constraints={"payment_send": {"amount": "<=1000"}},
        max_calls={"payment_send": 3},
    )
    sandbox = sandbox.sandbox

    print("""
Best practice: Use can_call() BEFORE attempting operations.
This avoids wasting compute and provides better error messages.
    """)

    # Demonstrate pre-flight checking
    test_amounts = [500, 1000, 1500, 100, 200, 300, 400]

    print("Pre-flight checks:")
    for amount in test_amounts:
        params = {"amount": amount}
        can = sandbox.can_call("payment_send", params)

        if can:
            # Actually make the call
            remaining = sandbox.get_remaining_calls("cap:method:payment_send")
            if remaining is not None and remaining > 0:
                sandbox.execute(f"await payment_send({{amount: {amount}}});")
                new_remaining = sandbox.get_remaining_calls("cap:method:payment_send")
                print(f"  ${amount}: ✓ Sent (remaining: {new_remaining})")
            else:
                print(f"  ${amount}: ✗ Budget exhausted")
        else:
            print(f"  ${amount}: ✗ Constraint violated (amount > 1000)")


# =============================================================================
# Part 7: Debugging Tips
# =============================================================================


def part7_debugging() -> None:
    """Tips for debugging sandbox issues."""
    print("\n" + "=" * 60)
    print("Part 7: Debugging Tips")
    print("=" * 60)

    print("""
DEBUGGING TIPS:

1. CHECK CAPABILITIES FIRST
   - Use sandbox.get_capabilities() to see what's allowed
   - Use sandbox.can_call(method, params) to test specific calls

2. INSPECT ERROR DETAILS
   - CapabilityError has method, params, and constraint info
   - JavaScript errors have stack traces

3. USE CONSOLE.LOG LIBERALLY
   - console.log() output becomes the return value
   - Great for debugging JavaScript logic

4. CHECK CALL BUDGETS
   - sandbox.get_call_counts() shows remaining calls
   - Budget exhaustion is a common issue

5. VALIDATE TOOL DEFINITIONS
   - Ensure parameter schemas are correct
   - Check that required fields are specified

6. ENABLE AUDIT LOGGING
   - AuditConfig captures all operations
   - Great for post-mortem debugging
    """)

    # Debugging example
    def process(data: str = "") -> dict[str, Any]:
        """Process data."""
        return {"processed": data}

    # Use create_sandbox_tool
    sandbox = create_sandbox_tool(tools=[process], max_calls={"process": 5})
    sandbox = sandbox.sandbox

    print("Debug inspection:")

    # Inspect capabilities
    print("\n  Capabilities:")
    for cap in sandbox.get_capabilities():
        print(f"    - {cap.method_pattern}")
        if cap.max_calls:
            print(f"      max_calls: {cap.max_calls}")
        if cap.constraints:
            print("      has constraints: yes")

    # Check call budgets
    print("\n  Call budgets:")
    for key, remaining in sandbox.get_call_counts().items():
        print(f"    {key}: {remaining} remaining")

    # Test specific calls
    print("\n  Authorization tests:")
    tests: list[tuple[str, dict[str, str]]] = [
        ("process", {"data": "hello"}),
        ("process", {}),  # Missing required field
        ("other_method", {}),  # No capability
    ]
    for method, params in tests:
        can = sandbox.can_call(method, params)
        status = "✓" if can else "✗"
        print(f"    {status} {method}({params})")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all tutorial parts."""
    print("=" * 60)
    print("Error Handling Tutorial")
    print("Capability Violations and Graceful Degradation")
    print("=" * 60)

    part1_error_types()
    part2_capability_violations()
    part3_js_errors()
    part4_tool_handler_errors()
    part5_graceful_degradation()
    part6_preflight_checks()
    part7_debugging()

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)
    print("""
Key takeaways:

1. Use can_call() for pre-flight authorization checks
2. Handle errors gracefully in JavaScript with try/catch
3. Tool handler exceptions propagate to JavaScript
4. Use patterns: retry, fallback, circuit breaker, Result type
5. Inspect capabilities and budgets for debugging

Error hierarchy:
  CapabilityError
    ├── CallLimitExceededError
    ├── ConstraintError
    └── MethodNotAllowedError

Best practices:
  - Always check can_call() before expensive operations
  - Use Result type pattern for cleaner error handling
  - Implement retries for transient failures
  - Log errors for debugging

Next tutorial: 07_rag_agent.py
    Build a RAG agent with vector search.
    """)


if __name__ == "__main__":
    main()
