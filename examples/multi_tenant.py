#!/usr/bin/env python3
"""Multi-Tenant Isolation: Secure tenant separation in shared environments.

Shows how to tenant-scoped capability enforcement.

Run: python multi_tenant.py
"""

from typing import Any
from dataclasses import dataclass

from amla_sandbox import (
    Sandbox,
    MethodCapability,
    ConstraintSet,
    Param,
    ToolDefinition,
    create_sandbox_tool,
)


@dataclass
class TenantContext:
    """Represents a tenant in the multi-tenant system."""

    tenant_id: str
    tenant_name: str
    plan: str  # "free", "pro", "enterprise"
    storage_quota_mb: int
    api_calls_per_hour: int


def main() -> None:
    # --- Setup: Multi-tenant tools ---
    tools = [
        ToolDefinition(
            name="read_document",
            description="Read a document from tenant storage",
            parameters={
                "type": "object",
                "properties": {
                    "tenant_id": {"type": "string"},
                    "document_id": {"type": "string"},
                },
                "required": ["tenant_id", "document_id"],
            },
        ),
        ToolDefinition(
            name="write_document",
            description="Write a document to tenant storage",
            parameters={
                "type": "object",
                "properties": {
                    "tenant_id": {"type": "string"},
                    "document_id": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["tenant_id", "document_id", "content"],
            },
        ),
        ToolDefinition(
            name="delete_document",
            description="Delete a document from tenant storage",
            parameters={
                "type": "object",
                "properties": {
                    "tenant_id": {"type": "string"},
                    "document_id": {"type": "string"},
                },
                "required": ["tenant_id", "document_id"],
            },
        ),
        ToolDefinition(
            name="query_database",
            description="Query the shared database",
            parameters={
                "type": "object",
                "properties": {
                    "tenant_id": {"type": "string"},
                    "table": {"type": "string"},
                    "filter": {"type": "string"},
                },
                "required": ["tenant_id", "table"],
            },
        ),
        ToolDefinition(
            name="call_external_api",
            description="Call an external API",
            parameters={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string"},
                    "method": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["endpoint", "method"],
            },
        ),
    ]
    tenant_documents: dict[str, dict[str, str]] = {
        "tenant-a": {"doc1": "Tenant A's document 1", "doc2": "Tenant A's document 2"},
        "tenant-b": {"doc1": "Tenant B's document 1"},
    }

    def handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
        tenant_id = params.get("tenant_id", "unknown")

        if method == "read_document":
            doc_id = params.get("document_id", "")
            docs = tenant_documents.get(tenant_id, {})
            content = docs.get(doc_id, "Document not found")
            return {"tenant_id": tenant_id, "document_id": doc_id, "content": content}

        elif method == "write_document":
            doc_id = params.get("document_id", "")
            content = params.get("content", "")
            if tenant_id not in tenant_documents:
                tenant_documents[tenant_id] = {}
            tenant_documents[tenant_id][doc_id] = content
            return {"tenant_id": tenant_id, "document_id": doc_id, "written": True}

        return {"success": True, "method": method, **params}

    # Basic Tenant Isolation
    print("--- Basic Tenant Isolation ---")

    def create_tenant_sandbox(tenant: TenantContext) -> Sandbox:
        """Create a sandbox scoped to a specific tenant."""
        return Sandbox(
            tools=tools,
            capabilities=[
                # Can only access own tenant's documents
                MethodCapability(
                    method_pattern="read_document",
                    constraints=ConstraintSet(
                        [
                            Param("tenant_id") == tenant.tenant_id,
                        ]
                    ),
                ),
                MethodCapability(
                    method_pattern="write_document",
                    constraints=ConstraintSet(
                        [
                            Param("tenant_id") == tenant.tenant_id,
                        ]
                    ),
                ),
                MethodCapability(
                    method_pattern="delete_document",
                    constraints=ConstraintSet(
                        [
                            Param("tenant_id") == tenant.tenant_id,
                        ]
                    ),
                ),
                MethodCapability(
                    method_pattern="query_database",
                    constraints=ConstraintSet(
                        [
                            Param("tenant_id") == tenant.tenant_id,
                        ]
                    ),
                ),
            ],
            tool_handler=handler,
        )

    # Create tenants
    tenant_a = TenantContext(
        tenant_id="tenant-a",
        tenant_name="Acme Corp",
        plan="pro",
        storage_quota_mb=1000,
        api_calls_per_hour=1000,
    )

    tenant_b = TenantContext(
        tenant_id="tenant-b",
        tenant_name="Beta Inc",
        plan="free",
        storage_quota_mb=100,
        api_calls_per_hour=100,
    )

    sandbox_a = create_tenant_sandbox(tenant_a)
    sandbox_b = create_tenant_sandbox(tenant_b)

    print(f"Tenant A ({tenant_a.tenant_name}):")
    print(
        f"  read own doc: {'allowed' if sandbox_a.can_call('read_document', {'tenant_id': 'tenant-a', 'document_id': 'doc1'}) else 'DENIED'}"
    )
    print(
        f"  read tenant-b doc: {'allowed' if sandbox_a.can_call('read_document', {'tenant_id': 'tenant-b', 'document_id': 'doc1'}) else 'DENIED'}"
    )

    print(f"\nTenant B ({tenant_b.tenant_name}):")
    print(
        f"  read own doc: {'allowed' if sandbox_b.can_call('read_document', {'tenant_id': 'tenant-b', 'document_id': 'doc1'}) else 'DENIED'}"
    )
    print(
        f"  read tenant-a doc: {'allowed' if sandbox_b.can_call('read_document', {'tenant_id': 'tenant-a', 'document_id': 'doc1'}) else 'DENIED'}"
    )

    # --- Example 2: Plan-Based Capabilities ---
    print("\n--- Plan-Based Feature Access ---")

    def create_tiered_sandbox(tenant: TenantContext) -> Sandbox:
        """Create sandbox with plan-based capabilities."""
        capabilities = [
            # All plans: basic document access
            MethodCapability(
                method_pattern="read_document",
                constraints=ConstraintSet(
                    [
                        Param("tenant_id") == tenant.tenant_id,
                    ]
                ),
            ),
            MethodCapability(
                method_pattern="write_document",
                constraints=ConstraintSet(
                    [
                        Param("tenant_id") == tenant.tenant_id,
                    ]
                ),
            ),
        ]

        # Pro and Enterprise: database queries
        if tenant.plan in ("pro", "enterprise"):
            capabilities.append(
                MethodCapability(
                    method_pattern="query_database",
                    constraints=ConstraintSet(
                        [
                            Param("tenant_id") == tenant.tenant_id,
                        ]
                    ),
                )
            )

        # Enterprise only: external API calls
        if tenant.plan == "enterprise":
            capabilities.append(
                MethodCapability(
                    method_pattern="call_external_api",
                )
            )

        # Enterprise only: delete (pro has soft-delete via support)
        if tenant.plan == "enterprise":
            capabilities.append(
                MethodCapability(
                    method_pattern="delete_document",
                    constraints=ConstraintSet(
                        [
                            Param("tenant_id") == tenant.tenant_id,
                        ]
                    ),
                )
            )

        return Sandbox(
            tools=tools,
            capabilities=capabilities,
            tool_handler=handler,
        )

    # Create tenant with enterprise plan
    enterprise_tenant = TenantContext(
        tenant_id="tenant-ent",
        tenant_name="Enterprise Co",
        plan="enterprise",
        storage_quota_mb=10000,
        api_calls_per_hour=10000,
    )

    free_sandbox = create_tiered_sandbox(tenant_b)  # tenant_b has free plan
    pro_sandbox = create_tiered_sandbox(tenant_a)  # tenant_a has pro plan
    enterprise_sandbox = create_tiered_sandbox(enterprise_tenant)

    print("Feature access by plan:")
    print("\n  Free plan (Tenant B):")
    print(
        f"    read_document: {'allowed' if free_sandbox.can_call('read_document', {'tenant_id': 'tenant-b', 'document_id': 'x'}) else 'DENIED'}"
    )
    print(
        f"    query_database: {'allowed' if free_sandbox.can_call('query_database', {'tenant_id': 'tenant-b', 'table': 'x'}) else 'DENIED'}"
    )
    print(
        f"    call_external_api: {'allowed' if free_sandbox.can_call('call_external_api', {'endpoint': '/api', 'method': 'GET'}) else 'DENIED'}"
    )

    print("\n  Pro plan (Tenant A):")
    print(
        f"    read_document: {'allowed' if pro_sandbox.can_call('read_document', {'tenant_id': 'tenant-a', 'document_id': 'x'}) else 'DENIED'}"
    )
    print(
        f"    query_database: {'allowed' if pro_sandbox.can_call('query_database', {'tenant_id': 'tenant-a', 'table': 'x'}) else 'DENIED'}"
    )
    print(
        f"    call_external_api: {'allowed' if pro_sandbox.can_call('call_external_api', {'endpoint': '/api', 'method': 'GET'}) else 'DENIED'}"
    )

    print("\n  Enterprise plan:")
    print(
        f"    read_document: {'allowed' if enterprise_sandbox.can_call('read_document', {'tenant_id': 'tenant-ent', 'document_id': 'x'}) else 'DENIED'}"
    )
    print(
        f"    query_database: {'allowed' if enterprise_sandbox.can_call('query_database', {'tenant_id': 'tenant-ent', 'table': 'x'}) else 'DENIED'}"
    )
    print(
        f"    call_external_api: {'allowed' if enterprise_sandbox.can_call('call_external_api', {'endpoint': '/api', 'method': 'GET'}) else 'DENIED'}"
    )
    print(
        f"    delete_document: {'allowed' if enterprise_sandbox.can_call('delete_document', {'tenant_id': 'tenant-ent', 'document_id': 'x'}) else 'DENIED'}"
    )

    # --- Example 3: Practical Multi-Tenant with create_sandbox_tool ---
    print("\n--- Multi-Tenant API with create_sandbox_tool ---")
    data_store: dict[str, dict[str, Any]] = {
        "tenant-1": {
            "users": [{"id": 1, "name": "Alice"}],
            "settings": {"theme": "light"},
        },
        "tenant-2": {
            "users": [{"id": 1, "name": "Bob"}],
            "settings": {"theme": "dark"},
        },
    }

    def get_users(tenant_id: str) -> dict[str, Any]:
        """Get users for a tenant."""
        tenant_data = data_store.get(tenant_id, {})
        return {"tenant_id": tenant_id, "users": tenant_data.get("users", [])}

    def get_settings(tenant_id: str) -> dict[str, Any]:
        """Get settings for a tenant."""
        tenant_data = data_store.get(tenant_id, {})
        return {"tenant_id": tenant_id, "settings": tenant_data.get("settings", {})}

    def update_settings(tenant_id: str, key: str, value: str) -> dict[str, Any]:
        """Update a setting for a tenant."""
        if tenant_id not in data_store:
            data_store[tenant_id] = {"users": [], "settings": {}}
        data_store[tenant_id]["settings"][key] = value
        return {"tenant_id": tenant_id, "updated": True, "key": key}

    def create_tenant_api(tenant_id: str) -> Any:
        """Create a tenant-scoped API."""
        return create_sandbox_tool(
            tools=[get_users, get_settings, update_settings],
            constraints={
                # All tools must use this tenant_id
                "get_users": {"tenant_id": [tenant_id]},
                "get_settings": {"tenant_id": [tenant_id]},
                "update_settings": {"tenant_id": [tenant_id]},
            },
            max_calls=100,
        )

    # Create tenant-scoped APIs
    tenant1_api = create_tenant_api("tenant-1")
    _tenant2_api = create_tenant_api("tenant-2")  # Available for testing

    print("Tenant 1 API:")
    result = tenant1_api.run(
        """
        // Can access own data
        const users = await get_users({tenant_id: "tenant-1"});
        console.log("Tenant 1 users:", users.users.length);

        // Update own settings
        await update_settings({tenant_id: "tenant-1", key: "theme", value: "dark"});
        console.log("Settings updated");

        // Try to access tenant-2 data (will fail)
        try {
            await get_users({tenant_id: "tenant-2"});
            console.log("Cross-tenant access: ALLOWED (bad!)");
        } catch (e) {
            console.log("Cross-tenant access: BLOCKED (correct!)");
        }
    """,
        language="javascript",
    )
    print(result)

    # Shared Resources with Scoped Access
    print("\n--- Shared Resources Pattern ---")

    # Some resources are shared but with tenant-specific views
    shared_templates = {
        "invoice": "Invoice template content",
        "report": "Report template content",
    }

    tenant_custom_templates: dict[str, dict[str, str]] = {
        "tenant-1": {"custom-header": "Tenant 1 Header"},
        "tenant-2": {"custom-header": "Tenant 2 Header"},
    }

    def get_template(template_id: str, tenant_id: str) -> dict[str, Any]:
        """Get a template (shared or tenant-specific)."""
        # Check tenant-specific first
        tenant_templates = tenant_custom_templates.get(tenant_id, {})
        if template_id in tenant_templates:
            return {
                "template_id": template_id,
                "content": tenant_templates[template_id],
                "source": "tenant-specific",
            }

        # Fall back to shared templates
        if template_id in shared_templates:
            return {
                "template_id": template_id,
                "content": shared_templates[template_id],
                "source": "shared",
            }

        return {"error": "Template not found"}

    def save_custom_template(
        template_id: str, content: str, tenant_id: str
    ) -> dict[str, Any]:
        """Save a tenant-specific template."""
        if tenant_id not in tenant_custom_templates:
            tenant_custom_templates[tenant_id] = {}
        tenant_custom_templates[tenant_id][template_id] = content
        return {"saved": True, "template_id": template_id, "tenant_id": tenant_id}

    shared_resource_sandbox = create_sandbox_tool(
        tools=[get_template, save_custom_template],
        constraints={
            # Templates can be read for any tenant (enforced by handler logic)
            # But saves must match the tenant
            "save_custom_template": {
                "tenant_id": ["tenant-1"],  # This API is for tenant-1
            },
        },
        max_calls=50,
    )

    print("Shared resource access:")
    result = shared_resource_sandbox.run(
        """
        // Read shared template
        const invoice = await get_template({template_id: "invoice", tenant_id: "tenant-1"});
        console.log("Invoice template source:", invoice.source);

        // Read tenant-specific template
        const header = await get_template({template_id: "custom-header", tenant_id: "tenant-1"});
        console.log("Header source:", header.source);

        // Save custom template (allowed for tenant-1)
        try {
            await save_custom_template({
                template_id: "custom-footer",
                content: "Tenant 1 Footer",
                tenant_id: "tenant-1"
            });
            console.log("Custom template save: SUCCESS");
        } catch (e) {
            console.log("Custom template save: BLOCKED");
        }

        // Try to save for tenant-2 (blocked)
        try {
            await save_custom_template({
                template_id: "custom-footer",
                content: "Sneaky content",
                tenant_id: "tenant-2"
            });
            console.log("Cross-tenant save: ALLOWED (bad!)");
        } catch (e) {
            console.log("Cross-tenant save: BLOCKED (correct!)");
        }
    """,
        language="javascript",
    )
    print(result)


if __name__ == "__main__":
    main()
