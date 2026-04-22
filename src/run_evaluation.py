"""
Exercise 04 Task 03 - Run agent evaluation via Python SDK.

Evaluates the handoff-service agent using direct domain accuracy scoring.
Supports both grounded data (from file) and synthetic data (built-in).

Usage:
    python src/run_evaluation.py                # Use synthetic test data
    python src/run_evaluation.py --grounded     # Use grounded data file
"""

import json
import os
import sys
import httpx
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Resolve paths relative to this script (works from any working directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

# --- Configuration ---
foundry_endpoint = os.environ["FOUNDRY_ENDPOINT"].rstrip("/")
model = os.environ["gpt_deployment"]
grounded_data_file = os.path.join(SCRIPT_DIR, "data", "handoff_service_evaluation_grounded.jsonl")
eval_output_file = os.path.join(SCRIPT_DIR, "data", "handoff_eval_results.jsonl")

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(credential, "https://ai.azure.com/.default")

# --- Synthetic test data ---
SYNTHETIC_DATA = [
    # Cart manager
    {"query": "Current domain: cora\nUser message: Add the blue paint to my cart", "expected_domain": "domain: cart_manager"},
    {"query": "Current domain: interior_designer\nUser message: Remove the rug from my cart please", "expected_domain": "domain: cart_manager"},
    {"query": "Current domain: cora\nUser message: I want to check out now", "expected_domain": "domain: cart_manager"},
    {"query": "Current domain: inventory_agent\nUser message: What's in my cart?", "expected_domain": "domain: cart_manager"},
    # Interior designer
    {"query": "Current domain: cora\nUser message: I need help choosing paint colors for my living room", "expected_domain": "domain: interior_designer"},
    {"query": "Current domain: cora\nUser message: Can you suggest furniture for a modern bedroom?", "expected_domain": "domain: interior_designer"},
    {"query": "Current domain: interior_designer\nUser message: What about a navy accent wall instead?", "expected_domain": "domain: interior_designer"},
    # Inventory
    {"query": "Current domain: cora\nUser message: Do you have PROD0018 in stock?", "expected_domain": "domain: inventory_agent"},
    {"query": "Current domain: interior_designer\nUser message: How many gallons of Deep Forest paint are available?", "expected_domain": "domain: inventory_agent"},
    {"query": "Current domain: cora\nUser message: Check inventory for paint rollers", "expected_domain": "domain: inventory_agent"},
    # Customer loyalty
    {"query": "Current domain: cora\nUser message: Do I have any discounts available? My ID is CUST001", "expected_domain": "domain: customer_loyalty"},
    {"query": "Current domain: cart_manager\nUser message: Am I eligible for a loyalty discount?", "expected_domain": "domain: customer_loyalty"},
    # Cora (general)
    {"query": "Current domain: cora\nUser message: Hello, what can you help me with?", "expected_domain": "domain: cora"},
    {"query": "Current domain: cora\nUser message: What are your store hours?", "expected_domain": "domain: cora"},
    {"query": "Current domain: interior_designer\nUser message: Actually, tell me more about Zava", "expected_domain": "domain: cora"},
    # Edge cases / ambiguous
    {"query": "Current domain: cora\nUser message: What kind of paints do you sell?", "expected_domain": "domain: cora"},
    {"query": "Current domain: interior_designer\nUser message: How much does that cost and is it in stock?", "expected_domain": "domain: inventory_agent"},
    {"query": "Current domain: cora\nUser message: I want to return a product", "expected_domain": "domain: cora"},
    {"query": "Current domain: cart_manager\nUser message: Actually I want design advice for my kitchen", "expected_domain": "domain: interior_designer"},
    {"query": "Current domain: cora\nUser message: Can I get a promo code?", "expected_domain": "domain: customer_loyalty"},
]


def call_handoff_agent(query: str) -> str:
    """Call the handoff-service agent via the Foundry Responses API."""
    token = token_provider()
    response = httpx.post(
        f"{foundry_endpoint}/openai/v1/responses",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json={
            "model": model,
            "input": query,
            "agent_reference": {"name": "handoff-service", "type": "agent_reference"},
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()

    for item in data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "")
    return data.get("output_text", str(data))


def extract_domain(response_text: str) -> str:
    """Extract the domain from the agent's JSON response."""
    try:
        parsed = json.loads(response_text)
        return parsed.get("domain", "")
    except (json.JSONDecodeError, TypeError):
        return ""


def extract_expected_domain(ground_truth: str) -> str:
    """Extract domain name from ground truth string like 'domain: cart_manager'."""
    return ground_truth.replace("domain: ", "").strip()


def load_data(use_grounded: bool) -> list[dict]:
    """Load test data from file (grounded) or use built-in synthetic data."""
    if use_grounded:
        print(f"Loading grounded data from {grounded_data_file}...")
        rows = []
        with open(grounded_data_file, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line.strip()))
        return rows
    else:
        print(f"Using {len(SYNTHETIC_DATA)} synthetic test cases...")
        return SYNTHETIC_DATA


def main():
    use_grounded = "--grounded" in sys.argv

    # --- Step 1: Load test data and call the agent ---
    rows = load_data(use_grounded)

    print(f"Running handoff-service agent on {len(rows)} test cases...\n")
    eval_data = []
    for i, row in enumerate(rows):
        query = row["query"]
        expected = row["expected_domain"]
        print(f"  [{i+1}/{len(rows)}] {query[:70]}...")

        try:
            response = call_handoff_agent(query)
        except Exception as e:
            print(f"    ERROR: {e}")
            response = f"error: {e}"

        eval_data.append({
            "query": query,
            "response": response,
            "ground_truth": expected,
        })

    # Save intermediate results
    with open(eval_output_file, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item) + "\n")
    print(f"\nAgent responses saved to {eval_output_file}")

    # --- Step 2: Evaluate domain accuracy ---
    print("\n" + "=" * 70)
    print("  HANDOFF-SERVICE EVALUATION RESULTS")
    print("=" * 70)

    correct = 0
    total = len(eval_data)
    domain_stats: dict[str, dict[str, int]] = {}

    print(f"\n{'#':>3}  {'Result':<6}  {'Expected':<22}  {'Predicted':<22}  {'Confidence'}")
    print("-" * 85)

    for i, item in enumerate(eval_data):
        expected_domain = extract_expected_domain(item["ground_truth"])
        predicted_domain = extract_domain(item["response"])

        # Parse confidence if available
        try:
            parsed = json.loads(item["response"])
            confidence = parsed.get("confidence", "N/A")
        except (json.JSONDecodeError, TypeError):
            confidence = "N/A"

        match = expected_domain == predicted_domain
        if match:
            correct += 1
        result_str = "PASS" if match else "FAIL"

        # Track per-domain stats
        if expected_domain not in domain_stats:
            domain_stats[expected_domain] = {"correct": 0, "total": 0}
        domain_stats[expected_domain]["total"] += 1
        if match:
            domain_stats[expected_domain]["correct"] += 1

        print(f"  {i+1:>2}  {result_str:<6}  {expected_domain:<22}  {predicted_domain:<22}  {confidence}")

    # --- Step 3: Summary ---
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n{'=' * 70}")
    print(f"  OVERALL ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
    print(f"{'=' * 70}")

    print(f"\n  Per-domain breakdown:")
    print(f"  {'Domain':<22}  {'Correct':>8}  {'Total':>6}  {'Accuracy':>9}")
    print(f"  {'-' * 50}")
    for domain in sorted(domain_stats.keys()):
        stats = domain_stats[domain]
        domain_acc = stats["correct"] / stats["total"] * 100
        print(f"  {domain:<22}  {stats['correct']:>8}  {stats['total']:>6}  {domain_acc:>8.1f}%")

    print()


if __name__ == "__main__":
    main()
