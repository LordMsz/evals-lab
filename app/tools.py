"""Fake backend 'tool' the agent can call."""

ORDERS = {
    "ORD-1001": {"status": "shipped", "carrier": "DHL", "expected_delivery": "2026-09-18"},
    "ORD-1002": {"status": "processing", "carrier": None, "expected_delivery": None},
    "ORD-1003": {"status": "delivered", "carrier": "UPS", "expected_delivery": "2026-09-10"},
}

TOOLS_SPEC = """Available tools (call at most one):
- lookup_order(order_id: str) -> order status, carrier, expected_delivery
"""


def lookup_order(order_id: str) -> dict:
    return ORDERS.get(order_id.upper(), {"error": f"order {order_id} not found"})


REGISTRY = {"lookup_order": lookup_order}
