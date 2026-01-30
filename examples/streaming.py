#!/usr/bin/env python3
"""Real-time Streaming Tutorial and Monitoring

This tutorial covers:
  1. Capturing console output in real-time
  2. Progress reporting patterns
  3. Monitoring tool call activity
  4. Building an observability dashboard
  5. Production monitoring best practices

Prerequisites:
    uv pip install "git+https://github.com/amlalabs/amla-sandbox"

Run:
    python streaming_and_monitoring.py

Time: ~15 minutes
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from amla_sandbox import create_sandbox_tool

# =============================================================================
# Part 1: Console Output Streaming
# =============================================================================


def part1_console_streaming() -> None:
    """Capture console.log output from the sandbox."""
    print("\n" + "=" * 60)
    print("Part 1: Console Output Streaming")
    print("=" * 60)

    print("""
Console output from JavaScript becomes the return value of run()/execute().
Multiple console.log() calls are concatenated with newlines.

For real-time streaming, process output incrementally or use progress callbacks.
    """)

    sandbox = create_sandbox_tool()

    # Basic output capture
    print("Basic output capture:")
    result = sandbox.run(
        """
        console.log("Step 1: Initializing...");
        console.log("Step 2: Processing...");
        console.log("Step 3: Complete!");
    """,
        language="javascript",
    )

    # Process line by line
    for i, line in enumerate(result.strip().split("\n")):
        print(f"  [{i + 1}] {line}")


# =============================================================================
# Part 2: Progress Reporting
# =============================================================================


def part2_progress_reporting() -> None:
    """Implement progress reporting patterns."""
    print("\n" + "=" * 60)
    print("Part 2: Progress Reporting")
    print("=" * 60)

    # Progress callback via tool
    progress_updates: list[dict[str, Any]] = []

    def report_progress(
        step: int, total: int, message: str, details: dict[str, Any] | None = None
    ) -> dict[str, bool]:
        """Report progress from the sandbox."""
        update = {
            "step": step,
            "total": total,
            "percent": round(step / total * 100, 1),
            "message": message,
            "details": details or {},
            "timestamp": time.time(),
        }
        progress_updates.append(update)
        print(f"  [{update['percent']:5.1f}%] {message}")
        return {"acknowledged": True}

    sandbox = create_sandbox_tool(
        tools=[report_progress],
    )

    print("Running task with progress reporting:\n")

    result = sandbox.run(
        """
        const items = ["users", "orders", "products", "analytics", "reports"];
        const total = items.length;

        await report_progress({step: 0, total, message: "Starting data export..."});

        for (let i = 0; i < items.length; i++) {
            // Simulate processing
            const item = items[i];
            await report_progress({
                step: i + 1,
                total,
                message: `Exported ${item}`,
                details: {table: item, rows: Math.floor(Math.random() * 1000) + 100}
            });
        }

        console.log("Export complete!");
    """,
        language="javascript",
    )

    print(f"\n{result.strip()}")
    print(f"\nTotal progress updates received: {len(progress_updates)}")


# =============================================================================
# Part 3: Tool Call Monitoring
# =============================================================================


def part3_tool_monitoring() -> None:
    """Monitor and log all tool calls."""
    print("\n" + "=" * 60)
    print("Part 3: Tool Call Monitoring")
    print("=" * 60)

    @dataclass
    class ToolCallMetrics:
        """Track metrics for tool calls."""

        total_calls: int = 0
        by_method: dict[str, int] = field(default_factory=lambda: dict[str, int]())
        durations: list[float] = field(default_factory=lambda: list[float]())
        errors: int = 0

    metrics = ToolCallMetrics()

    def data_fetch(id: int = 0) -> dict[str, Any]:
        """Fetch data by ID."""
        start = time.time()
        metrics.total_calls += 1
        metrics.by_method["data_fetch"] = metrics.by_method.get("data_fetch", 0) + 1
        result = {"id": id, "status": "ok", "processed_at": datetime.now().isoformat()}
        metrics.durations.append((time.time() - start) * 1000)
        return result

    def data_process(batch: int = 0) -> dict[str, Any]:
        """Process a data batch."""
        start = time.time()
        metrics.total_calls += 1
        metrics.by_method["data_process"] = metrics.by_method.get("data_process", 0) + 1
        result = {
            "batch": batch,
            "status": "ok",
            "processed_at": datetime.now().isoformat(),
        }
        metrics.durations.append((time.time() - start) * 1000)
        return result

    def data_store(final: bool = False) -> dict[str, Any]:
        """Store processed data."""
        start = time.time()
        metrics.total_calls += 1
        metrics.by_method["data_store"] = metrics.by_method.get("data_store", 0) + 1
        result = {
            "final": final,
            "status": "ok",
            "processed_at": datetime.now().isoformat(),
        }
        metrics.durations.append((time.time() - start) * 1000)
        return result

    sandbox = create_sandbox_tool(
        tools=[data_fetch, data_process, data_store],
    )

    print("Running operations with monitoring:\n")

    sandbox.run(
        """
        // Run some operations
        for (let i = 0; i < 5; i++) {
            await data_fetch({id: i});
        }

        for (let i = 0; i < 3; i++) {
            await data_process({batch: i});
        }

        await data_store({final: true});
    """,
        language="javascript",
    )

    # Report metrics
    print("Tool Call Metrics:")
    print(f"  Total calls: {metrics.total_calls}")
    print("\n  By method:")
    for method, count in sorted(metrics.by_method.items()):
        print(f"    {method}: {count} calls")

    if metrics.durations:
        avg_duration = sum(metrics.durations) / len(metrics.durations)
        print(f"\n  Average duration: {avg_duration:.2f}ms")
        print(f"  Max duration: {max(metrics.durations):.2f}ms")


# =============================================================================
# Part 4: Execution Tracing
# =============================================================================


def part4_execution_tracing() -> None:
    """Trace execution flow through the sandbox."""
    print("\n" + "=" * 60)
    print("Part 4: Execution Tracing")
    print("=" * 60)

    @dataclass
    class TraceEvent:
        """A single event in the execution trace."""

        timestamp: float
        event_type: str
        method: str | None = None
        details: dict[str, Any] | None = None

    trace: list[TraceEvent] = []

    def make_traced_fn(name: str) -> Any:
        """Create a traced function with the given name."""

        def traced_fn(**kwargs: Any) -> dict[str, Any]:
            trace.append(
                TraceEvent(
                    time.time(), "call_start", name, {"params": list(kwargs.keys())}
                )
            )
            result = {"method": name, "ok": True}
            trace.append(TraceEvent(time.time(), "call_end", name, {"success": True}))
            return result

        traced_fn.__name__ = name
        traced_fn.__doc__ = f"Execute {name}"
        return traced_fn

    step1 = make_traced_fn("step1")
    step2 = make_traced_fn("step2")
    step3 = make_traced_fn("step3")

    sandbox = create_sandbox_tool(
        tools=[step1, step2, step3],
    )

    # Mark execution start
    trace.append(TraceEvent(time.time(), "execution_start"))

    sandbox.run(
        """
        console.log("Starting workflow...");
        await step1({data: "initial"});
        await step2({transform: true});
        await step3({finalize: true});
        console.log("Workflow complete!");
    """,
        language="javascript",
    )

    # Mark execution end
    trace.append(TraceEvent(time.time(), "execution_end"))

    print("Execution trace:")
    start_time = trace[0].timestamp
    for event in trace:
        elapsed = (event.timestamp - start_time) * 1000
        if event.method:
            print(f"  [{elapsed:6.1f}ms] {event.event_type}: {event.method}")
        else:
            print(f"  [{elapsed:6.1f}ms] {event.event_type}")


# =============================================================================
# Part 5: Health Checks and Status
# =============================================================================


def part5_health_checks() -> None:
    """Implement health checks for sandbox operations."""
    print("\n" + "=" * 60)
    print("Part 5: Health Checks and Status")
    print("=" * 60)

    import random

    @dataclass
    class ServiceHealth:
        """Track service health status."""

        name: str
        healthy: bool = True
        last_check: float = 0
        consecutive_failures: int = 0
        total_calls: int = 0
        total_failures: int = 0

    services: dict[str, ServiceHealth] = {
        "database": ServiceHealth("database"),
        "cache": ServiceHealth("cache"),
        "external_api": ServiceHealth("external_api"),
    }

    def _check_service(service_name: str) -> dict[str, Any]:
        """Check a service and track health."""
        service = services.get(service_name)
        if service:
            service.total_calls += 1
            service.last_check = time.time()
            if random.random() < 0.2:  # 20% failure rate for demo
                service.consecutive_failures += 1
                service.total_failures += 1
                if service.consecutive_failures >= 3:
                    service.healthy = False
                raise ConnectionError(f"{service_name} temporarily unavailable")
            service.consecutive_failures = 0
            service.healthy = True
        return {"status": "ok"}

    def database_query(query_id: int) -> dict[str, Any]:
        """Query the database."""
        return _check_service("database")

    def cache_get(key: str) -> dict[str, Any]:
        """Get from cache."""
        return _check_service("cache")

    def external_api_call(endpoint: str) -> dict[str, Any]:
        """Call external API."""
        return _check_service("external_api")

    sandbox = create_sandbox_tool(
        tools=[database_query, cache_get, external_api_call],
        max_calls=100,
    )

    print("Running operations and tracking health:\n")

    sandbox.run(
        """
        for (let i = 0; i < 10; i++) {
            try { await database_query({query_id: i}); } catch {}
            try { await cache_get({key: "user_" + i}); } catch {}
            try { await external_api_call({endpoint: "/status"}); } catch {}
        }
    """,
        language="javascript",
    )

    print("Service Health Report:")
    print("-" * 50)
    for name, health in services.items():
        status = "HEALTHY" if health.healthy else "UNHEALTHY"
        success_rate = (
            (health.total_calls - health.total_failures) / health.total_calls * 100
            if health.total_calls > 0
            else 0
        )
        print(f"  {name}:")
        print(f"    Status: {status}")
        print(f"    Total calls: {health.total_calls}")
        print(f"    Failures: {health.total_failures}")
        print(f"    Success rate: {success_rate:.1f}%")


# =============================================================================
# Part 6: Observability Dashboard
# =============================================================================


def part6_observability_dashboard() -> None:
    """Build a simple observability dashboard."""
    print("\n" + "=" * 60)
    print("Part 6: Observability Dashboard")
    print("=" * 60)

    import random

    @dataclass
    class DashboardMetrics:
        """Aggregated metrics for dashboard."""

        start_time: float = field(default_factory=time.time)
        request_count: int = 0
        error_count: int = 0
        latencies: list[float] = field(default_factory=lambda: list[float]())
        methods: dict[str, int] = field(default_factory=lambda: dict[str, int]())
        active_operations: int = 0

        def record_request(
            self, method: str, latency_ms: float, error: bool = False
        ) -> None:
            self.request_count += 1
            self.latencies.append(latency_ms)
            self.methods[method] = self.methods.get(method, 0) + 1
            if error:
                self.error_count += 1

        def get_summary(self) -> dict[str, Any]:
            uptime = time.time() - self.start_time
            avg_latency = (
                sum(self.latencies) / len(self.latencies) if self.latencies else 0
            )
            p99_latency = (
                sorted(self.latencies)[int(len(self.latencies) * 0.99)]
                if len(self.latencies) >= 10
                else avg_latency
            )

            return {
                "uptime_seconds": round(uptime, 2),
                "total_requests": self.request_count,
                "error_rate": round(self.error_count / self.request_count * 100, 2)
                if self.request_count > 0
                else 0,
                "avg_latency_ms": round(avg_latency, 2),
                "p99_latency_ms": round(p99_latency, 2),
                "requests_per_second": round(self.request_count / uptime, 2)
                if uptime > 0
                else 0,
            }

    dashboard = DashboardMetrics()

    def _record_api_call(method_name: str) -> dict[str, Any]:
        """Record an API call with metrics."""
        start = time.time()
        dashboard.active_operations += 1
        try:
            time.sleep(random.uniform(0.001, 0.01))
            latency = (time.time() - start) * 1000
            dashboard.record_request(method_name, latency)
            return {"method": method_name, "ok": True}
        finally:
            dashboard.active_operations -= 1

    def api_users(action: str) -> dict[str, Any]:
        """User operations."""
        return _record_api_call("api_users")

    def api_orders(action: str) -> dict[str, Any]:
        """Order operations."""
        return _record_api_call("api_orders")

    def api_products(action: str) -> dict[str, Any]:
        """Product operations."""
        return _record_api_call("api_products")

    sandbox = create_sandbox_tool(
        tools=[api_users, api_orders, api_products],
        max_calls=200,
    )

    print("Running workload for dashboard metrics...\n")

    sandbox.run(
        """
        for (let i = 0; i < 50; i++) {
            await api_users({action: "list"});
            if (i % 2 === 0) await api_orders({action: "create"});
            if (i % 3 === 0) await api_products({action: "search"});
        }
    """,
        language="javascript",
    )

    # Print dashboard
    summary = dashboard.get_summary()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║              OBSERVABILITY DASHBOARD                     ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(
        f"║  Uptime:              {summary['uptime_seconds']:>8.2f} seconds              ║"
    )
    print(
        f"║  Total Requests:      {summary['total_requests']:>8}                       ║"
    )
    print(
        f"║  Requests/sec:        {summary['requests_per_second']:>8.2f}                       ║"
    )
    print(
        f"║  Error Rate:          {summary['error_rate']:>8.2f}%                      ║"
    )
    print(
        f"║  Avg Latency:         {summary['avg_latency_ms']:>8.2f} ms                    ║"
    )
    print(
        f"║  P99 Latency:         {summary['p99_latency_ms']:>8.2f} ms                    ║"
    )
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Requests by Endpoint:                                   ║")
    for method, count in sorted(dashboard.methods.items(), key=lambda x: -x[1]):
        bar = "█" * min(count // 2, 30)
        print(f"║    {method:<15} {count:>5} {bar:<30}║")
    print("╚══════════════════════════════════════════════════════════╝")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run all tutorial parts."""
    print("=" * 60)
    print("Real-time Streaming Tutorial and Monitoring")
    print("=" * 60)

    part1_console_streaming()
    part2_progress_reporting()
    part3_tool_monitoring()
    part4_execution_tracing()
    part5_health_checks()
    part6_observability_dashboard()

    print("\n" + "=" * 60)
    print("Tutorial Complete!")
    print("=" * 60)
    print("""
Key takeaways:

1. Console output is captured via run()/execute() return value
2. Use tool callbacks for progress reporting
3. Wrap handlers for metrics collection
4. Trace execution with start/end events
5. Track service health with consecutive failure counts
6. Build dashboards from aggregated metrics

Production recommendations:
  - Export metrics to Prometheus/Datadog
  - Use structured logging (JSON)
  - Implement distributed tracing
  - Set up alerting on error rates
  - Monitor resource usage (memory, CPU)

Congratulations! You've completed all tutorials!

For more examples, see:
  - examples/00_bash_tool.py - Progressive disclosure layers
  - examples/08_insurance_claims.py - Real-world use case
  - examples/agents/ - Complete agent implementations
    """)


if __name__ == "__main__":
    main()
