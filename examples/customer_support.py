#!/usr/bin/env python3
"""Customer Support Agent: Automated ticket handling with constraints.

In this example:
- Real-world customer support automation
- Capability-based access control for sensitive operations
- Multi-step ticket resolution
- Audit logging for compliance

Run: python customer_support.py
"""

from typing import Any
from datetime import datetime

from amla_sandbox import (
    create_sandbox_tool,
)


# --- Customer Support Tools ---


def get_customer(customer_id: str) -> dict[str, Any]:
    """Get customer information.

    Args:
        customer_id: Customer ID
    """
    customers = {
        "C001": {
            "id": "C001",
            "name": "Alice Johnson",
            "email": "alice@example.com",
            "tier": "gold",
            "since": "2022-01-15",
        },
        "C002": {
            "id": "C002",
            "name": "Bob Smith",
            "email": "bob@example.com",
            "tier": "standard",
            "since": "2023-06-20",
        },
    }
    return customers.get(customer_id, {"error": "Customer not found"})


def get_orders(customer_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """Get customer's recent orders.

    Args:
        customer_id: Customer ID
        limit: Maximum orders to return
    """
    orders = {
        "C001": [
            {
                "id": "O001",
                "date": "2024-01-15",
                "total": 149.99,
                "status": "delivered",
            },
            {"id": "O002", "date": "2024-01-20", "total": 89.99, "status": "shipped"},
        ],
        "C002": [
            {
                "id": "O003",
                "date": "2024-01-18",
                "total": 299.99,
                "status": "processing",
            },
        ],
    }
    return orders.get(customer_id, [])[:limit]


def issue_refund(order_id: str, amount: float, reason: str) -> dict[str, Any]:
    """Issue a refund for an order.

    Args:
        order_id: Order ID to refund
        amount: Refund amount
        reason: Reason for refund
    """
    return {
        "success": True,
        "refund_id": f"REF-{order_id}",
        "amount": amount,
        "reason": reason,
        "processed_at": datetime.now().isoformat(),
    }


def apply_discount(
    customer_id: str, discount_percent: int, reason: str
) -> dict[str, Any]:
    """Apply a discount to customer's next order.

    Args:
        customer_id: Customer ID
        discount_percent: Discount percentage (1-100)
        reason: Reason for discount
    """
    return {
        "success": True,
        "customer_id": customer_id,
        "discount": f"{discount_percent}%",
        "code": f"DISC-{customer_id}-{discount_percent}",
        "reason": reason,
    }


def escalate_ticket(
    ticket_id: str, reason: str, priority: str = "normal"
) -> dict[str, Any]:
    """Escalate a ticket to human support.

    Args:
        ticket_id: Ticket ID
        reason: Reason for escalation
        priority: Priority level (low, normal, high, urgent)
    """
    return {
        "escalated": True,
        "ticket_id": ticket_id,
        "reason": reason,
        "priority": priority,
        "assigned_team": "tier2-support",
    }


def send_email(to: str, subject: str, body: str) -> dict[str, bool]:
    """Send an email to the customer.

    Args:
        to: Recipient email
        subject: Email subject
        body: Email body
    """
    print(f"  [EMAIL] To: {to}")
    print(f"  [EMAIL] Subject: {subject}")
    return {"sent": True}


def main() -> None:
    # --- Level 1 Support Agent (Limited Capabilities) ---
    print("=== Level 1 Support Agent ===")
    print("Can: View customer data, small refunds (<$50), send emails")
    print("Cannot: Large refunds, discounts, escalations\n")

    level1_sandbox = create_sandbox_tool(
        tools=[get_customer, get_orders, issue_refund, send_email],
        constraints={
            "issue_refund": {
                "amount": "<=50",  # Max $50 refund
                "reason": ["damaged_item", "late_delivery", "wrong_item"],
            },
        },
        max_calls={
            "issue_refund": 5,  # Max 5 refunds per session
            "send_email": 10,
        },
    )

    # Simulate ticket handling
    result = level1_sandbox.run(
        """
        // Ticket: Customer C001 reports damaged item, wants refund

        // Step 1: Get customer info
        const customer = await get_customer({customer_id: "C001"});
        console.log("Customer:", customer.name, "(", customer.tier, "tier )");

        // Step 2: Get order history
        const orders = await get_orders({customer_id: "C001"});
        const recentOrder = orders[0];
        console.log("Recent order:", recentOrder.id, "- $" + recentOrder.total);

        // Step 3: Issue refund (within our limit)
        if (recentOrder.total <= 50) {
            const refund = await issue_refund({
                order_id: recentOrder.id,
                amount: recentOrder.total,
                reason: "damaged_item"
            });
            console.log("Refund issued:", refund.refund_id);
        } else {
            console.log("Order total exceeds L1 refund limit ($50)");
            console.log("Would need to escalate to L2 support");
        }

        // Step 4: Send confirmation email
        await send_email({
            to: customer.email,
            subject: "Your refund request",
            body: "We're processing your refund request..."
        });
    """,
        language="javascript",
    )
    print(result)

    # --- Level 2 Support Agent (More Capabilities) ---
    print("\n=== Level 2 Support Agent ===")
    print("Can: Larger refunds (<$500), discounts (<20%), escalations\n")

    level2_sandbox = create_sandbox_tool(
        tools=[
            get_customer,
            get_orders,
            issue_refund,
            apply_discount,
            escalate_ticket,
            send_email,
        ],
        constraints={
            "issue_refund": {
                "amount": "<=500",  # Higher limit
            },
            "apply_discount": {
                "discount_percent": "<=20",  # Max 20% discount
            },
            "escalate_ticket": {
                "priority": ["low", "normal", "high"],  # Can't set urgent
            },
        },
        max_calls={
            "issue_refund": 10,
            "apply_discount": 5,
        },
    )

    result = level2_sandbox.run(
        """
        // Ticket: Customer C001 unhappy with service, high-value order issue

        const customer = await get_customer({customer_id: "C001"});
        const orders = await get_orders({customer_id: "C001"});

        console.log("Handling escalated ticket for:", customer.name);

        // Process refund for damaged order
        const refund = await issue_refund({
            order_id: orders[0].id,
            amount: 149.99,
            reason: "customer_satisfaction"
        });
        console.log("Refund processed:", refund.amount);

        // Apply courtesy discount for inconvenience
        const discount = await apply_discount({
            customer_id: customer.id,
            discount_percent: 15,
            reason: "service_recovery"
        });
        console.log("Discount applied:", discount.code);

        // Send resolution email
        await send_email({
            to: customer.email,
            subject: "Issue resolved - with a thank you",
            body: "We've processed your refund and added a 15% discount..."
        });

        console.log("\\nTicket resolved with refund + discount");
    """,
        language="javascript",
    )
    print(result)

    # --- Constraint Violation Example ---
    print("\n=== Constraint Enforcement Example ===")

    result = level1_sandbox.run(
        """
        // L1 agent tries to issue a large refund

        try {
            const refund = await issue_refund({
                order_id: "O003",
                amount: 299.99,  // Exceeds $50 limit
                reason: "damaged_item"
            });
            console.log("Refund issued");
        } catch (e) {
            console.log("Refund blocked: amount exceeds L1 authority ($50 max)");
        }

        // Try invalid reason
        try {
            const refund = await issue_refund({
                order_id: "O001",
                amount: 25,
                reason: "customer_demanded"  // Not an allowed reason
            });
        } catch (e) {
            console.log("Refund blocked: reason not in allowed list");
        }
    """,
        language="javascript",
    )
    print(result)

    # --- Multi-Step Resolution Flow ---
    print("\n=== Multi-Step Resolution Flow ===")

    result = level2_sandbox.run(
        """
        // Complex ticket: Multiple issues to resolve

        const customerId = "C001";
        const customer = await get_customer({customer_id: customerId});
        const orders = await get_orders({customer_id: customerId});

        console.log("=== Ticket Resolution Summary ===");
        console.log("Customer:", customer.name);
        console.log("Tier:", customer.tier);
        console.log("\\nActions taken:");

        // 1. Process partial refund
        const refund = await issue_refund({
            order_id: orders[0].id,
            amount: 50.00,
            reason: "partial_refund"
        });
        console.log("1. Partial refund: $" + refund.amount);

        // 2. Apply loyalty discount
        const discount = await apply_discount({
            customer_id: customerId,
            discount_percent: 10,
            reason: "loyalty_appreciation"
        });
        console.log("2. Discount applied:", discount.discount);

        // 3. Save resolution notes to VFS
        const notes = {
            resolved_at: new Date().toISOString(),
            customer_id: customerId,
            actions: ["partial_refund", "loyalty_discount"],
            resolution: "Customer satisfied with partial refund and future discount"
        };
        await fs.writeFile('/state/ticket_resolution.json', JSON.stringify(notes, null, 2));
        console.log("3. Resolution notes saved");

        // 4. Send final email
        await send_email({
            to: customer.email,
            subject: "Your support ticket has been resolved",
            body: "Thank you for your patience..."
        });
        console.log("4. Confirmation email sent");

        console.log("\\n=== Ticket Closed ===");
    """,
        language="javascript",
    )
    print(result)


if __name__ == "__main__":
    main()
