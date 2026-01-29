#!/usr/bin/env python3
"""Data Pipeline Agent: ETL operations with sandbox isolation.

Shows extract-transform-load (ETL) patterns using sandbox tools.

Run: python data_pipeline.py
"""

from typing import Any

from amla_sandbox import create_sandbox_tool


def fetch_source_data(source: str, batch_size: int = 100) -> dict[str, Any]:
    """Fetch data from a source system."""
    if source == "sales":
        return {
            "source": source,
            "records": [
                {"id": f"S{i}", "product": f"P{i % 5}", "amount": 100 + i * 10}
                for i in range(min(batch_size, 50))
            ],
            "total_available": 1000,
        }
    elif source == "customers":
        return {
            "source": source,
            "records": [
                {
                    "id": f"C{i}",
                    "name": f"Customer {i}",
                    "tier": ["gold", "silver", "bronze"][i % 3],
                }
                for i in range(min(batch_size, 30))
            ],
            "total_available": 500,
        }
    return {"source": source, "records": [], "total_available": 0}


def validate_record(record: dict[str, Any], schema: str) -> dict[str, Any]:
    """Validate a record against a schema."""
    errors: list[str] = []
    if schema == "sales":
        if "amount" in record and record["amount"] < 0:
            errors.append("amount cannot be negative")
        if "product" not in record:
            errors.append("product is required")
    return {"valid": len(errors) == 0, "record": record, "errors": errors}


def load_to_warehouse(table: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Load records to the data warehouse."""
    return {"success": True, "table": table, "records_loaded": len(records)}


def main() -> None:
    sandbox = create_sandbox_tool(
        tools=[fetch_source_data, validate_record, load_to_warehouse],
        max_calls=200,
    )

    # Example 1: Simple ETL Pipeline
    print("--- Simple ETL Pipeline ---")
    result = sandbox.run(
        """
        console.log("ETL Pipeline\\n");
        const data = await fetch_source_data({source: "sales", batch_size: 5});
        console.log("Extracted:", data.records.length, "records");

        const valid = [];
        for (const r of data.records) {
            const v = await validate_record({record: r, schema: "sales"});
            if (v.valid) valid.push({...r, processed: true});
        }
        console.log("Validated:", valid.length, "records");

        const load = await load_to_warehouse({table: "fact_sales", records: valid});
        console.log("Loaded:", load.records_loaded, "to", load.table);
    """,
        language="javascript",
    )
    print(result)

    # Example 2: Multi-Source Pipeline
    print("\n--- Multi-Source Pipeline ---")
    result = sandbox.run(
        """
        console.log("Multi-source ETL\\n");
        const results = {};
        for (const src of ["sales", "customers"]) {
            const data = await fetch_source_data({source: src, batch_size: 10});
            const load = await load_to_warehouse({table: "stg_" + src, records: data.records});
            results[src] = load.records_loaded;
            console.log(src + ":", load.records_loaded, "loaded");
        }
        console.log("\\nTotal sources:", Object.keys(results).length);
    """,
        language="javascript",
    )
    print(result)

    # Example 3: Batched Processing with VFS Checkpoint
    print("\n--- Batched Processing ---")
    result = sandbox.run(
        """
        const data = await fetch_source_data({source: "sales", batch_size: 12});
        let processed = 0;
        const batchSize = 4;

        for (let i = 0; i < data.records.length; i += batchSize) {
            const batch = data.records.slice(i, i + batchSize);
            await load_to_warehouse({table: "fact_sales", records: batch});
            processed += batch.length;
            await fs.writeFile("/workspace/checkpoint.json", JSON.stringify({processed}));
            console.log("Batch", Math.floor(i/batchSize)+1, "- total:", processed);
        }
        console.log("\\nFinal checkpoint:", await fs.readFile("/workspace/checkpoint.json"));
    """,
        language="javascript",
    )
    print(result)

    # Example 4: Data Quality Check
    print("\n--- Data Quality Check ---")
    result = sandbox.run(
        """
        const data = await fetch_source_data({source: "sales", batch_size: 10});
        let valid = 0, total = 0, sum = 0;
        for (const r of data.records) {
            const v = await validate_record({record: r, schema: "sales"});
            if (v.valid) valid++;
            total++;
            sum += r.amount;
        }
        console.log("Quality Report:");
        console.log("  Records:", total);
        console.log("  Valid:", valid, "(" + Math.round(valid/total*100) + "%)");
        console.log("  Total Amount: $" + sum);
    """,
        language="javascript",
    )
    print(result)


if __name__ == "__main__":
    main()
