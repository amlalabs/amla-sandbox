#!/usr/bin/env python3
"""Insurance Claims Demo - Capability-Based Multi-Agent Security

This demonstrates the insurance claims workflow from the blog post, showing:

1. Multi-agent workflow with capability attenuation
2. Budget enforcement (max_calls limits)
3. Constraint validation at each step
4. Attack prevention (unauthorized agent rejection)

The Workflow:
    Customer → Claims Agent → Payout Agent → Gateway

    Each step attenuates capabilities:
    - Customer: "I claim up to $25,000"
    - Claims Agent: "Damage assessed at $3,800"
    - Payout Agent: "Execute $3,300 payout (after deductible)"
    - Gateway: "Verified and paid"
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from amla_sandbox import (
    ConstraintSet,
    MethodCapability,
    Param,
    Sandbox,
    ToolDefinition,
)


# =============================================================================
# Domain Types
# =============================================================================


@dataclass
class Claim:
    """An insurance claim."""

    claim_id: str
    customer_id: str
    policy_max: int  # Max allowed by policy (cents)
    description: str


@dataclass
class Assessment:
    """Damage assessment result."""

    claim_id: str
    assessed_amount: int  # Assessed damage (cents)
    deductible: int  # Deductible (cents)
    payout_amount: int  # After deductible


# =============================================================================
# Mock Backend (In production, these would be real APIs)
# =============================================================================


class MockInsuranceBackend:
    """Simulates the insurance company's backend systems."""

    def __init__(self) -> None:
        self.claims: dict[str, Claim] = {}
        self.assessments: dict[str, Assessment] = {}
        self.payouts: dict[str, int] = {}
        self.call_log: list[str] = []

    def policy_lookup(self, customer_id: str) -> dict[str, Any]:
        """Look up customer's policy."""
        self.call_log.append(f"policy.lookup({customer_id})")
        return {
            "customer_id": customer_id,
            "policy_type": "auto",
            "max_claim": 2500000,  # $25,000 in cents
            "deductible": 50000,  # $500 deductible
        }

    def claims_create(
        self, customer_id: str, description: str, amount: int
    ) -> dict[str, Any]:
        """Create a new claim."""
        self.call_log.append(f"claims.create({customer_id}, amount={amount})")
        claim_id = f"CLM-{len(self.claims) + 1:06d}"
        policy = self.policy_lookup(customer_id)

        # Enforce policy max
        if amount > policy["max_claim"]:
            raise ValueError(
                f"Amount {amount} exceeds policy max {policy['max_claim']}"
            )

        claim = Claim(
            claim_id=claim_id,
            customer_id=customer_id,
            policy_max=policy["max_claim"],
            description=description,
        )
        self.claims[claim_id] = claim
        return {
            "claim_id": claim_id,
            "status": "created",
            "policy_max": policy["max_claim"],
        }

    def claims_assess(self, claim_id: str, assessed_amount: int) -> dict[str, Any]:
        """Record damage assessment."""
        self.call_log.append(f"claims.assess({claim_id}, amount={assessed_amount})")
        claim = self.claims.get(claim_id)
        if not claim:
            raise ValueError(f"Claim {claim_id} not found")

        policy = self.policy_lookup(claim.customer_id)
        deductible = policy["deductible"]
        payout = max(0, assessed_amount - deductible)

        assessment = Assessment(
            claim_id=claim_id,
            assessed_amount=assessed_amount,
            deductible=deductible,
            payout_amount=payout,
        )
        self.assessments[claim_id] = assessment
        return {
            "claim_id": claim_id,
            "assessed": assessed_amount,
            "deductible": deductible,
            "payout_approved": payout,
        }

    def payout_execute(self, claim_id: str, amount: int) -> dict[str, Any]:
        """Execute the payout."""
        self.call_log.append(f"payout.execute({claim_id}, amount={amount})")
        assessment = self.assessments.get(claim_id)
        if not assessment:
            raise ValueError(f"No assessment for claim {claim_id}")
        if amount > assessment.payout_amount:
            raise ValueError(
                f"Payout {amount} exceeds approved {assessment.payout_amount}"
            )

        self.payouts[claim_id] = amount
        return {"claim_id": claim_id, "paid": amount, "status": "completed"}


# =============================================================================
# Tool Handler
# =============================================================================

ToolHandler = Callable[[str, dict[str, Any]], dict[str, Any]]


def create_tool_handler(backend: MockInsuranceBackend) -> ToolHandler:
    """Create tool handler that routes to backend."""

    def handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
        # Route based on method name (mcp:toolname format)
        tool_name = method.replace("mcp:", "")

        if tool_name == "policy.lookup":
            return backend.policy_lookup(params["customer_id"])
        elif tool_name == "claims.create":
            return backend.claims_create(
                params["customer_id"], params["description"], params["amount"]
            )
        elif tool_name == "claims.assess":
            return backend.claims_assess(params["claim_id"], params["assessed_amount"])
        elif tool_name == "payout.execute":
            return backend.payout_execute(params["claim_id"], params["amount"])
        else:
            raise ValueError(f"Unknown tool: {method}")

    return handler


# =============================================================================
# Tool Definitions
# =============================================================================

TOOLS = [
    ToolDefinition(
        name="policy.lookup",
        description="Look up customer policy details",
        parameters={
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    ),
    ToolDefinition(
        name="claims.create",
        description="Create a new insurance claim",
        parameters={
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "description": {"type": "string"},
                "amount": {"type": "integer", "description": "Claim amount in cents"},
            },
            "required": ["customer_id", "description", "amount"],
        },
    ),
    ToolDefinition(
        name="claims.assess",
        description="Record damage assessment for a claim",
        parameters={
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "assessed_amount": {
                    "type": "integer",
                    "description": "Assessed damage in cents",
                },
            },
            "required": ["claim_id", "assessed_amount"],
        },
    ),
    ToolDefinition(
        name="payout.execute",
        description="Execute a payout for a claim",
        parameters={
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
                "amount": {"type": "integer", "description": "Payout amount in cents"},
            },
            "required": ["claim_id", "amount"],
        },
    ),
]


# =============================================================================
# Capability Definitions (Different for each agent role)
# =============================================================================


def make_claims_agent_caps() -> list[MethodCapability]:
    """Capabilities for the Claims Agent.

    Can:
    - Look up policies
    - Create claims (up to policy max)
    - Assess claims
    - NOT execute payouts
    """
    return [
        # Read-only policy access
        MethodCapability(
            method_pattern="mcp:policy.lookup",
            max_calls=100,  # Budget for lookups
        ),
        # Create claims with amount constraint
        MethodCapability(
            method_pattern="mcp:claims.create",
            constraints=ConstraintSet(
                [
                    Param("amount") <= 2500000,  # Max $25,000
                    Param("amount") > 0,
                ]
            ),
            max_calls=50,  # Budget for claim creation
        ),
        # Assess claims
        MethodCapability(
            method_pattern="mcp:claims.assess",
            constraints=ConstraintSet(
                [
                    Param("assessed_amount")
                    <= 2500000,  # Can't assess more than policy max
                    Param("assessed_amount") > 0,
                ]
            ),
            max_calls=50,
        ),
        # NOTE: No payout.execute capability!
    ]


def make_payout_agent_caps(max_payout: int) -> list[MethodCapability]:
    """Capabilities for the Payout Agent.

    Can ONLY:
    - Execute payouts up to the attenuated max
    """
    return [
        MethodCapability(
            method_pattern="mcp:payout.execute",
            constraints=ConstraintSet(
                [
                    Param("amount") <= max_payout,  # Attenuated limit!
                    Param("amount") > 0,
                ]
            ),
            max_calls=1,  # Only ONE payout per delegation
        ),
    ]


def make_attacker_caps() -> list[MethodCapability]:
    """Eve's forged capabilities (won't work!)"""
    return [
        # Eve tries to give herself payout rights
        MethodCapability(
            method_pattern="mcp:payout.execute",
            constraints=ConstraintSet([Param("amount") <= 10000000]),  # $100k!
            max_calls=999,
        ),
    ]


# =============================================================================
# Demo: The Happy Path
# =============================================================================


def run_happy_path(backend: MockInsuranceBackend) -> None:
    """Demonstrate the normal claims workflow."""
    print("=" * 70)
    print("ACT 1: THE HAPPY PATH")
    print("=" * 70)
    print()
    print("Customer files a claim -> Claims Agent processes -> Payout executes")
    print()

    # Step 1: Claims Agent processes the claim
    print("--- Claims Agent ---")
    claims_agent = Sandbox(
        tools=TOOLS,
        capabilities=make_claims_agent_caps(),
        tool_handler=create_tool_handler(backend),
    )

    print("  Capabilities:")
    for cap in claims_agent.get_capabilities():
        remaining = claims_agent.get_remaining_calls(cap.key())
        budget = f"budget: {remaining}" if remaining else "unlimited"
        print(f"    - {cap.method_pattern} ({budget})")

    # Agent creates and assesses the claim
    result = claims_agent.execute("""
        // Customer: "My car was damaged, claiming $3,800"
        const claim = await claims_create({
            customer_id: "CUST-001",
            description: "Fender bender in parking lot",
            amount: 380000  // $3,800 in cents
        });
        console.log("Claim created:", JSON.stringify(claim));

        // Assess the damage
        const assessment = await claims_assess({
            claim_id: claim.claim_id,
            assessed_amount: 380000
        });
        console.log("Assessment:", JSON.stringify(assessment));
    """)
    print(f"  {result.strip()}")
    print()

    # Get assessment from the backend directly (in production, would be passed via PCA chain)

    claim_id = list(backend.assessments.keys())[0]
    assessment = {
        "claim_id": claim_id,
        "payout_approved": backend.assessments[claim_id].payout_amount,
    }
    payout_amount = int(assessment["payout_approved"])

    # Step 2: Payout Agent (with attenuated capability)
    print("--- Payout Agent ---")
    print(f"  Delegated with max payout: ${payout_amount / 100:.2f}")

    payout_agent = Sandbox(
        tools=TOOLS,
        capabilities=make_payout_agent_caps(payout_amount),  # Attenuated!
        tool_handler=create_tool_handler(backend),
    )

    print("  Capabilities:")
    for cap in payout_agent.get_capabilities():
        remaining = payout_agent.get_remaining_calls(cap.key())
        print(f"    - {cap.method_pattern} (budget: {remaining})")
        if cap.constraints:
            print(f"      constraints: amount <= ${payout_amount / 100:.2f}")

    result = payout_agent.execute(f"""
        const result = await payout_execute({{
            claim_id: "{assessment["claim_id"]}",
            amount: {payout_amount}
        }});
        console.log("Payout result:", JSON.stringify(result));
    """)
    print(f"  {result.strip()}")
    print()

    print("--- Result ---")
    print("  Claim processed successfully!")
    claim_id = str(assessment["claim_id"])
    print(f"  Payout: ${backend.payouts[claim_id] / 100:.2f}")
    print()


# =============================================================================
# Demo: Attack Scenarios
# =============================================================================


def run_attack_scenarios(backend: MockInsuranceBackend) -> None:
    """Demonstrate attacks that FAIL due to capability enforcement."""
    print("=" * 70)
    print("ACT 2: ATTACKS THAT FAIL")
    print("=" * 70)
    print()

    # Attack 1: Claims Agent tries to execute payout directly
    print("--- Attack 1: Claims Agent Tries Direct Payout ---")
    print("  Claims Agent says: 'I'll just pay myself directly!'")
    print()

    claims_agent = Sandbox(
        tools=TOOLS,
        capabilities=make_claims_agent_caps(),  # No payout capability!
        tool_handler=create_tool_handler(backend),
    )

    # Check if agent can even call payout
    can_payout = claims_agent.can_call("mcp:payout.execute", {"amount": 100000})
    print(f"  can_call('payout.execute')? {can_payout}")

    if not can_payout:
        print("  BLOCKED: Claims Agent has no payout.execute capability!")
    print()

    # Attack 2: Payout Agent tries to exceed limit
    print("--- Attack 2: Payout Agent Tries to Exceed Limit ---")
    print("  Payout Agent says: 'Why not $50,000 instead of $3,300?'")
    print()

    payout_agent = Sandbox(
        tools=TOOLS,
        capabilities=make_payout_agent_caps(330000),  # Max $3,300
        tool_handler=create_tool_handler(backend),
    )

    can_high_payout = payout_agent.can_call("mcp:payout.execute", {"amount": 5000000})
    print(f"  can_call(amount=$50,000)? {can_high_payout}")

    if not can_high_payout:
        print("  BLOCKED: Constraint 'amount <= $3,300' not satisfied!")
    print()

    # Attack 3: Payout Agent tries second payout (budget exhausted)
    print("--- Attack 3: Payout Agent Double-Dips ---")
    print("  Payout Agent says: 'Let me process another payout!'")
    print()

    payout_agent = Sandbox(
        tools=TOOLS,
        capabilities=make_payout_agent_caps(330000),  # max_calls=1
        tool_handler=create_tool_handler(backend),
    )

    print(
        f"  Initial budget: {payout_agent.get_remaining_calls('cap:method:mcp:payout.execute')}"
    )

    # First payout succeeds
    try:
        result = payout_agent.execute("""
            const r = await payout_execute({claim_id: "CLM-000001", amount: 330000});
            console.log("First payout:", JSON.stringify(r));
        """)
        print(f"  {result.strip()}")
    except Exception as e:
        print(f"  First payout error: {e}")

    print(
        f"  Budget after first: {payout_agent.get_remaining_calls('cap:method:mcp:payout.execute')}"
    )

    # Second payout fails
    can_second = payout_agent.can_call("mcp:payout.execute", {"amount": 100000})
    print(f"  can_call(second payout)? {can_second}")

    if not can_second:
        print("  BLOCKED: Budget exhausted (max_calls=1)!")
    print()

    # Attack 4: Eve intercepts and tries with forged capabilities
    print("--- Attack 4: Eve with Forged Capabilities ---")
    print("  Eve says: 'I have the chain, I'll just forge new capabilities!'")
    print()

    # Eve creates her own sandbox with fake capabilities
    _eve_sandbox = Sandbox(
        tools=TOOLS,
        capabilities=make_attacker_caps(),  # Eve's fake caps
        tool_handler=create_tool_handler(backend),
    )

    # Even if Eve has matching patterns, the backend will reject
    # because the PCA chain (pca= parameter) would be invalid
    # For now, we show the capability layer blocks unauthorized patterns

    # Let's also check if there's a matching claim_id
    print("  Eve has capability for payout.execute with amount <= $100k")
    print("  BUT: In real system, PCA signature verification would fail")
    print("  The capability pattern matches, but the cryptographic chain is invalid")
    print()
    print("  In production:")
    print("    1. Gateway verifies PCA signature chain")
    print("    2. Checks designated_executor matches Eve's public key")
    print("    3. MISMATCH: Eve is not the designated continuation")
    print("    4. REJECTED: 'Signer is not the designated continuation'")
    print()


# =============================================================================
# Demo: Introspection for DX
# =============================================================================


def run_dx_demo() -> None:
    """Show developer experience features."""
    print("=" * 70)
    print("ACT 3: DEVELOPER EXPERIENCE")
    print("=" * 70)
    print()

    backend = MockInsuranceBackend()

    sandbox = Sandbox(
        tools=TOOLS,
        capabilities=[
            MethodCapability(method_pattern="mcp:policy.lookup", max_calls=100),
            MethodCapability(
                method_pattern="mcp:claims.create",
                constraints=ConstraintSet([Param("amount") <= 2500000]),
                max_calls=50,
            ),
            MethodCapability(method_pattern="mcp:claims.assess", max_calls=50),
        ],
        tool_handler=create_tool_handler(backend),
    )

    # 1. Introspect capabilities
    print("--- 1. List All Capabilities ---")
    for cap in sandbox.get_capabilities():
        print(f"  {cap.key()}")
        if cap.max_calls:
            print(f"    max_calls: {cap.max_calls}")
        if cap.constraints:
            print(f"    constraints: {len(cap.constraints)} rules")
    print()

    # 2. Check budgets
    print("--- 2. Check Budgets ---")
    budgets = sandbox.get_call_counts()
    for key, remaining in budgets.items():
        print(f"  {key}: {remaining} remaining")
    print()

    # 3. Pre-flight authorization check
    print("--- 3. Pre-flight Authorization Checks ---")
    checks = [
        ("mcp:policy.lookup", {"customer_id": "C123"}),
        (
            "mcp:claims.create",
            {"customer_id": "C123", "description": "Test", "amount": 100000},
        ),
        (
            "mcp:claims.create",
            {"customer_id": "C123", "description": "Test", "amount": 9999999},
        ),
        ("mcp:payout.execute", {"claim_id": "X", "amount": 1000}),
    ]

    for method, params in checks:
        allowed = sandbox.can_call(method, params)
        status = "ALLOWED" if allowed else "DENIED"
        print(f"  {method}({params})")
        print(f"    -> {status}")
    print()

    # 4. Budget consumption tracking
    print("--- 4. Budget Tracking After Operations ---")
    print(
        f"  Before: policy.lookup budget = {sandbox.get_remaining_calls('cap:method:mcp:policy.lookup')}"
    )

    sandbox.execute('await policy_lookup({customer_id: "CUST-001"});')
    sandbox.execute('await policy_lookup({customer_id: "CUST-002"});')
    sandbox.execute('await policy_lookup({customer_id: "CUST-003"});')

    print(
        f"  After 3 lookups: budget = {sandbox.get_remaining_calls('cap:method:mcp:policy.lookup')}"
    )
    print()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the complete insurance claims demo."""
    print()
    print("INSURANCE CLAIMS DEMO")
    print("Capability-Based Multi-Agent Security")
    print("=" * 70)
    print()

    backend = MockInsuranceBackend()

    run_happy_path(backend)
    run_attack_scenarios(backend)
    run_dx_demo()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key Security Properties Demonstrated:

1. LEAST PRIVILEGE
   - Claims Agent can't execute payouts
   - Payout Agent can only pay up to approved amount

2. ATTENUATION
   - Policy max ($25k) -> Assessed ($3.8k) -> Payout ($3.3k)
   - Each step can only narrow, never widen

3. BUDGET ENFORCEMENT
   - Payout Agent gets exactly 1 payout attempt
   - Double-dipping blocked by max_calls

4. CONSTRAINT VALIDATION
   - Amount limits enforced at capability layer
   - No way to bypass via parameter manipulation

5. INTROSPECTION
   - Agents can check what they're allowed to do
   - Pre-flight checks avoid wasted attempts
   - Budget visibility for planning

In Production:
   - Add PCA cryptographic chain verification
   - Gateway validates designated_executor
   - Full audit trail from chain signatures
""")


if __name__ == "__main__":
    main()
