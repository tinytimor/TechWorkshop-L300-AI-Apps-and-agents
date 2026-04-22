"""
Exercise 04 - Per-Agent Quality Evaluation

Evaluates each specialized Foundry agent (cora, inventory, interior-designer,
cart-manager, customer-loyalty) with targeted queries and scores response
quality using keyword relevance, format checks, and F1 scoring.

Usage:
    python src/run_agent_quality_eval.py                  # All agents
    python src/run_agent_quality_eval.py --agent cora     # Single agent
    python src/run_agent_quality_eval.py --upload         # Upload results to Foundry portal
"""

import json
import os
import sys
import time
import httpx
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# --- Path setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

# --- Configuration ---
foundry_endpoint = os.environ["FOUNDRY_ENDPOINT"].rstrip("/")
model = os.environ["gpt_deployment"]

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(credential, "https://ai.azure.com/.default")

# --- Test cases per agent ---
AGENT_TEST_CASES = {
    "cora": {
        "display_name": "Cora (Shopper Assistant)",
        "cases": [
            {
                "query": "Hello, what can you help me with?",
                "expected_keywords": ["help", "paint", "product", "shop", "zava", "assist", "design"],
                "description": "Greeting and capabilities",
            },
            {
                "query": "What types of products does Zava sell?",
                "expected_keywords": ["paint", "product", "sprayer", "accessor", "shade", "color"],
                "description": "Product catalog overview",
            },
            {
                "query": "Tell me about your store",
                "expected_keywords": ["zava", "home", "improve", "paint"],
                "description": "Store information",
            },
            {
                "query": "I'm repainting my house and don't know where to start. Can you guide me?",
                "expected_keywords": ["paint", "color", "help", "room", "recommend"],
                "description": "Open-ended guidance request",
            },
            {
                "query": "Do you offer any deals right now?",
                "expected_keywords": ["discount", "deal", "offer", "loyalty", "promotion", "save"],
                "description": "Promotions inquiry",
            },
            {
                "query": "What's the difference between interior and exterior paint?",
                "expected_keywords": ["interior", "exterior", "paint", "durable", "finish"],
                "description": "Product knowledge question",
            },
        ],
    },
    "inventory-agent": {
        "display_name": "Inventory Agent",
        "cases": [
            {
                "query": "Check if PROD0001 is in stock",
                "expected_keywords": ["PROD0001", "stock", "avail", "inventory", "unit"],
                "description": "Specific product stock check",
            },
            {
                "query": "Do you have any paint sprayers available?",
                "expected_keywords": ["sprayer", "stock", "avail", "PROD"],
                "description": "Category inventory check",
            },
            {
                "query": "How many gallons of Midnight Blue paint are in stock?",
                "expected_keywords": ["midnight", "blue", "stock", "avail", "gallon"],
                "description": "Named product availability",
            },
            {
                "query": "Check inventory for PROD0033 and PROD0045",
                "expected_keywords": ["PROD0033", "PROD0045", "stock", "avail", "inventory"],
                "description": "Multiple product IDs check",
            },
            {
                "query": "Is the Deep Forest paint shade available?",
                "expected_keywords": ["deep forest", "stock", "avail", "paint"],
                "description": "Product by name lookup",
            },
            {
                "query": "Do you have drop cloths in stock?",
                "expected_keywords": ["drop cloth", "stock", "avail", "accessor"],
                "description": "Accessory availability check",
            },
        ],
    },
    "cart-manager": {
        "display_name": "Cart Manager",
        "cases": [
            {
                "query": "Add 2 units of PROD0001 to my cart",
                "expected_keywords": ["cart", "added", "PROD0001"],
                "description": "Add item to cart",
                "expect_json_fields": ["cart", "answer"],
            },
            {
                "query": "Show me what's in my cart",
                "expected_keywords": ["cart"],
                "description": "View cart contents",
                "expect_json_fields": ["cart"],
            },
            {
                "query": "Remove PROD0001 from my cart",
                "expected_keywords": ["remov", "cart", "PROD0001"],
                "description": "Remove item from cart",
                "expect_json_fields": ["cart", "answer"],
            },
            {
                "query": "Change the quantity of PROD0001 to 5",
                "expected_keywords": ["cart", "PROD0001", "5", "updat", "quantit"],
                "description": "Update item quantity",
                "expect_json_fields": ["cart", "answer"],
            },
            {
                "query": "Clear my entire cart",
                "expected_keywords": ["cart", "clear", "empty"],
                "description": "Clear cart",
                "expect_json_fields": ["cart", "answer"],
            },
            {
                "query": "I'd like to check out now please",
                "expected_keywords": ["checkout", "order", "Miami", "pickup", "Zava"],
                "description": "Checkout flow (expects Miami store reference)",
                "expect_json_fields": ["cart", "answer"],
            },
        ],
    },
    "interior-designer": {
        "display_name": "Interior Designer",
        "cases": [
            {
                "query": "I need help choosing paint colors for a modern living room with grey furniture",
                "expected_keywords": ["color", "paint", "living", "modern", "recommend", "grey", "gray"],
                "description": "Color scheme recommendation",
            },
            {
                "query": "What paint finish should I use for a kitchen?",
                "expected_keywords": ["finish", "kitchen", "paint", "semi-gloss", "satin", "durable"],
                "description": "Finish type advice",
            },
            {
                "query": "I have a small bedroom with low light. What colors would make it feel bigger?",
                "expected_keywords": ["light", "bright", "color", "paint", "small", "room", "white"],
                "description": "Space-enhancing color advice",
            },
            {
                "query": "Can you suggest accessories to go with a navy blue accent wall?",
                "expected_keywords": ["navy", "blue", "accent", "accessor", "complement", "recommend"],
                "description": "Complementary accessories upsell",
            },
            {
                "query": "What's trending in home paint colors this year?",
                "expected_keywords": ["trend", "color", "paint", "popular", "style"],
                "description": "Trend awareness",
            },
            {
                "query": "I want to redo my bathroom. Should I use matte or glossy paint?",
                "expected_keywords": ["bathroom", "matte", "gloss", "finish", "moisture", "paint"],
                "description": "Room-specific finish recommendation",
            },
        ],
    },
    "customer-loyalty": {
        "display_name": "Customer Loyalty",
        "cases": [
            {
                "query": "Check my loyalty status and available discounts. Customer ID: CUST001",
                "expected_keywords": ["discount", "loyalty", "CUST001", "%"],
                "description": "Loyalty status and discount check",
            },
            {
                "query": "Am I eligible for any promotions?",
                "expected_keywords": ["customer", "id", "provide", "need"],
                "description": "Discount request without customer ID (should ask for ID)",
            },
            {
                "query": "My customer ID is CUST001. What tier am I in?",
                "expected_keywords": ["CUST001", "tier", "discount", "loyal"],
                "description": "Tier inquiry with customer ID",
            },
            {
                "query": "How do I earn more loyalty points? Customer ID: CUST001",
                "expected_keywords": ["loyalty", "point", "earn", "CUST001", "discount"],
                "description": "Loyalty program engagement",
            },
        ],
    },
}


def call_agent(agent_name: str, query: str) -> str:
    """Call a Foundry agent via the Responses API."""
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
            "agent_reference": {"name": agent_name, "type": "agent_reference"},
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


def compute_f1(predicted_tokens: set, expected_tokens: set) -> float:
    """Compute F1 score between predicted and expected token sets."""
    if not predicted_tokens or not expected_tokens:
        return 0.0
    common = predicted_tokens & expected_tokens
    if not common:
        return 0.0
    precision = len(common) / len(predicted_tokens)
    recall = len(common) / len(expected_tokens)
    return 2 * (precision * recall) / (precision + recall)


def evaluate_response(response: str, test_case: dict) -> dict:
    """Evaluate a single response against expected characteristics."""
    scores = {}

    # 1. Response presence (non-empty, meaningful length)
    scores["has_response"] = 1.0 if len(response.strip()) > 20 else 0.0

    # 2. Keyword coverage (case-insensitive partial matching)
    keywords = test_case.get("expected_keywords", [])
    if keywords:
        response_lower = response.lower()
        found = sum(1 for kw in keywords if kw.lower() in response_lower)
        scores["keyword_coverage"] = round(found / len(keywords), 2)
    else:
        scores["keyword_coverage"] = 1.0

    # 3. F1 score (token-level overlap with expected keywords)
    response_tokens = set(response.lower().split())
    expected_tokens = set(kw.lower() for kw in keywords)
    scores["f1_score"] = round(compute_f1(response_tokens, expected_tokens), 2)

    # 4. JSON format validation (for structured agents)
    expected_fields = test_case.get("expect_json_fields")
    if expected_fields:
        try:
            parsed = json.loads(response)
            scores["valid_json"] = 1.0
            fields_found = sum(1 for f in expected_fields if f in parsed)
            scores["json_fields"] = round(fields_found / len(expected_fields), 2)
        except (json.JSONDecodeError, TypeError):
            scores["valid_json"] = 0.0
            scores["json_fields"] = 0.0

    # 5. Error-free response
    error_indicators = ["error", "exception", "traceback", "failed to", "i cannot"]
    scores["no_error"] = 0.0 if any(e in response.lower() for e in error_indicators) else 1.0

    # 6. Response length quality (between 50 and 5000 chars is ideal)
    length = len(response.strip())
    if 50 <= length <= 5000:
        scores["length_quality"] = 1.0
    elif 20 <= length < 50 or 5000 < length <= 10000:
        scores["length_quality"] = 0.5
    else:
        scores["length_quality"] = 0.0

    # Overall weighted score
    score_values = list(scores.values())
    scores["overall"] = round(sum(score_values) / len(score_values), 2)

    return scores


def main():
    # Parse args
    target_agent = None
    if "--agent" in sys.argv:
        idx = sys.argv.index("--agent")
        if idx + 1 < len(sys.argv):
            target_agent = sys.argv[idx + 1]

    agents_to_test = {target_agent: AGENT_TEST_CASES[target_agent]} if target_agent else AGENT_TEST_CASES

    total_cases = sum(len(v["cases"]) for v in agents_to_test.values())
    print("=" * 80)
    print("  PER-AGENT QUALITY EVALUATION")
    print("=" * 80)
    print(f"  Agents: {', '.join(agents_to_test.keys())}")
    print(f"  Total test cases: {total_cases}")
    print(f"  Endpoint: {foundry_endpoint}")
    print("=" * 80)

    all_results = []
    agent_summaries = {}

    for agent_name, agent_config in agents_to_test.items():
        display = agent_config["display_name"]
        cases = agent_config["cases"]

        print(f"\n{'─' * 80}")
        print(f"  Agent: {display} ({agent_name})")
        print(f"{'─' * 80}")

        agent_scores = []

        for i, test_case in enumerate(cases):
            query = test_case["query"]
            desc = test_case["description"]
            print(f"\n  [{i+1}/{len(cases)}] {desc}")
            print(f"       Query: {query[:70]}{'...' if len(query) > 70 else ''}")

            start_time = time.time()
            try:
                response = call_agent(agent_name, query)
                elapsed = time.time() - start_time
            except Exception as e:
                print(f"       ERROR: {e}")
                response = f"error: {e}"
                elapsed = time.time() - start_time

            # Truncate response for display
            response_preview = response[:120].replace("\n", " ")
            print(f"       Response: {response_preview}{'...' if len(response) > 120 else ''}")
            print(f"       Latency: {elapsed:.1f}s")

            # Evaluate
            scores = evaluate_response(response, test_case)
            overall = scores["overall"]
            kw = scores["keyword_coverage"]
            f1 = scores["f1_score"]

            status = "PASS" if overall >= 0.6 else "WARN" if overall >= 0.4 else "FAIL"
            print(f"       Score: {overall:.0%} ({status})  |  Keywords: {kw:.0%}  |  F1: {f1:.2f}")

            agent_scores.append(overall)
            all_results.append({
                "agent": agent_name,
                "description": desc,
                "query": query,
                "response": response,
                "scores": scores,
                "latency_s": round(elapsed, 2),
            })

        avg = sum(agent_scores) / len(agent_scores) if agent_scores else 0
        agent_summaries[agent_name] = {
            "display_name": display,
            "avg_score": avg,
            "num_cases": len(cases),
        }

    # --- Summary ---
    print("\n" + "=" * 80)
    print("  EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\n  {'Agent':<30}  {'Cases':>6}  {'Avg Score':>10}  {'Status'}")
    print(f"  {'─' * 65}")

    overall_scores = []
    for agent_name, summary in agent_summaries.items():
        avg = summary["avg_score"]
        status = "PASS" if avg >= 0.6 else "WARN" if avg >= 0.4 else "FAIL"
        print(f"  {summary['display_name']:<30}  {summary['num_cases']:>6}  {avg:>9.0%}  {status}")
        overall_scores.append(avg)

    grand_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    print(f"\n  {'─' * 65}")
    print(f"  {'OVERALL':30}  {total_cases:>6}  {grand_avg:>9.0%}")
    print("=" * 80)

    # Save detailed results
    output_file = os.path.join(SCRIPT_DIR, "data", "agent_quality_eval_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Detailed results saved to {output_file}")

    # --- Upload to Foundry portal ---
    if "--upload" in sys.argv:
        upload_to_foundry(all_results)
    else:
        print("  Tip: Add --upload to push results to the Foundry portal\n")


def upload_to_foundry(all_results: list[dict]):
    """Upload evaluation results to the NEW Azure AI Foundry portal.

    Uses the azure-ai-projects Cloud Evaluation API (server-side evaluators).
    This populates the AI Quality metrics columns (Coherence, Relevance, Fluency)
    and makes results visible in the new Foundry Evaluations tab.
    """
    print(f"\n{'─' * 80}")
    print("  Uploading to Azure AI Foundry (Cloud Evaluation API)...")
    print(f"{'─' * 80}")

    try:
        from azure.ai.projects import AIProjectClient
        from openai.types.eval_create_params import DataSourceConfigCustom
        from openai.types.evals.create_eval_jsonl_run_data_source_param import (
            CreateEvalJSONLRunDataSourceParam,
            SourceFileContent,
            SourceFileContentContent,
        )
    except ImportError as e:
        print(f"  ERROR: Missing SDK package: {e}")
        print("    pip install 'azure-ai-projects>=2.0.0'")
        return

    # --- Build inline data from results ---
    inline_items = []
    for row in all_results:
        ground_truth = ", ".join(
            next(
                (tc.get("expected_keywords", [])
                 for agent_cfg in AGENT_TEST_CASES.values()
                 for tc in agent_cfg["cases"]
                 if tc["query"] == row["query"]),
                []
            )
        )
        inline_items.append(SourceFileContentContent(item={
            "query": row["query"],
            "response": row["response"],
            "ground_truth": ground_truth,
        }))

    # --- Data schema ---
    data_source_config = DataSourceConfigCustom(
        type="custom",
        item_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "response": {"type": "string"},
                "ground_truth": {"type": "string"},
            },
            "required": ["query", "response"],
        },
        include_sample_schema=True,
    )

    # --- Testing criteria: Azure AI built-in evaluators as raw dicts ---
    # The azure_ai_evaluator type is an Azure extension not in the OpenAI types,
    # so we pass them as plain dicts which the API accepts.
    testing_criteria = [
        # AI Quality (AI-assisted, run server-side — populates AI Quality columns)
        {
            "type": "azure_ai_evaluator",
            "name": "coherence",
            "evaluator_name": "builtin.coherence",
            "initialization_parameters": {"deployment_name": model},
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "relevance",
            "evaluator_name": "builtin.relevance",
            "initialization_parameters": {"deployment_name": model},
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "fluency",
            "evaluator_name": "builtin.fluency",
            "initialization_parameters": {"deployment_name": model},
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            },
        },
        # Safety evaluators (no deployment_name needed, run via Content Safety)
        {
            "type": "azure_ai_evaluator",
            "name": "violence",
            "evaluator_name": "builtin.violence",
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "self_harm",
            "evaluator_name": "builtin.self_harm",
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            },
        },
        # NLP evaluator (token-level F1 against ground truth)
        {
            "type": "azure_ai_evaluator",
            "name": "f1_score",
            "evaluator_name": "builtin.f1_score",
            "data_mapping": {
                "response": "{{item.response}}",
                "ground_truth": "{{item.ground_truth}}",
            },
        },
    ]

    try:
        project_client = AIProjectClient(
            endpoint=foundry_endpoint,
            credential=credential,
        )
        client = project_client.get_openai_client()

        # Step 1: Create evaluation definition
        # Use extra_body to pass Azure AI evaluators since they're an Azure extension
        # not in the base OpenAI TestingCriterion types.
        print("  Creating evaluation definition...")
        eval_object = client.evals.create(
            name="Agent Quality Evaluation",
            data_source_config=data_source_config,
            testing_criteria=[],
            extra_body={"testing_criteria": testing_criteria},
        )
        print(f"  Eval ID: {eval_object.id}")

        # Step 2: Create a run with inline data
        print(f"  Submitting {len(inline_items)} test cases...")
        eval_run = client.evals.runs.create(
            eval_id=eval_object.id,
            name="agent-quality-run",
            data_source=CreateEvalJSONLRunDataSourceParam(
                type="jsonl",
                source=SourceFileContent(
                    type="file_content",
                    content=inline_items,
                ),
            ),
        )
        print(f"  Run ID: {eval_run.id}")

        # Step 3: Poll for completion
        import time as _time
        poll_count = 0
        while True:
            run = client.evals.runs.retrieve(
                run_id=eval_run.id, eval_id=eval_object.id
            )
            if run.status in ("completed", "failed", "canceled"):
                break
            poll_count += 1
            status_msg = f"  Waiting for server-side evaluation... ({run.status})"
            if poll_count % 6 == 0:
                status_msg += f" [{poll_count * 5}s elapsed]"
            print(status_msg)
            _time.sleep(5)

        if run.status == "completed":
            print(f"\n  Evaluation completed!")
            report_url = getattr(run, "report_url", None)
            if report_url:
                print(f"  View in Foundry portal: {report_url}")

            # Print result counts
            result_counts = getattr(run, "result_counts", None)
            if result_counts:
                print(f"\n  Result counts:")
                for key, value in vars(result_counts).items():
                    if not key.startswith("_") and value is not None:
                        print(f"    {key}: {value}")

            # Fetch per-row results
            try:
                output_items = list(client.evals.runs.output_items.list(
                    run_id=run.id, eval_id=eval_object.id
                ))
                if output_items:
                    print(f"\n  Per-row results ({len(output_items)} items):")
                    for item in output_items[:5]:
                        results = getattr(item, "results", None) or {}
                        scores_str = "  |  ".join(
                            f"{k}: {v}" for k, v in (
                                results if isinstance(results, dict) else {}
                            ).items()
                        )
                        print(f"    {scores_str[:100]}")
                    if len(output_items) > 5:
                        print(f"    ... and {len(output_items) - 5} more rows")
            except Exception:
                pass
        else:
            print(f"\n  Evaluation {run.status}.")
            error = getattr(run, "error", None)
            if error:
                print(f"  Error: {error}")

        print()

    except Exception as e:
        print(f"\n  Upload failed: {e}")
        print("  Local results are still available in:")
        print(f"    {os.path.join(SCRIPT_DIR, 'data', 'agent_quality_eval_results.json')}\n")


if __name__ == "__main__":
    main()
