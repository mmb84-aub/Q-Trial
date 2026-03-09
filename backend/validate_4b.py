import json

with open("outputs/agentic_run.json") as f:
    data = json.load(f)

rs = data.get("reasoning_state", {})
dispatched = rs.get("dispatched_tool_results", [])

print(f"dispatched_tool_results count: {len(dispatched)}")
for i, r in enumerate(dispatched):
    req = r.get("request", {})
    print(f"\n[dispatched[{i}]]")
    print(f"  tool_type   : {req.get('tool_type')}")
    print(f"  hypothesis  : {req.get('hypothesis_id')}")
    print(f"  columns     : {req.get('columns')}")
    print(f"  group_col   : {req.get('group_column')}")
    print(f"  priority    : {req.get('priority')}")
    print(f"  tool_called : {r.get('tool_called')}")
    print(f"  error       : {r.get('error')}")
    result = r.get("result")
    if result:
        keys = list(result.keys())[:6]
        print(f"  result keys : {keys}")
        # Print a sample value for first key
        first_key = keys[0]
        val = result[first_key]
        if isinstance(val, (str, int, float, bool)):
            print(f"  sample      : {first_key} = {val}")
        elif isinstance(val, dict):
            print(f"  sample      : {first_key} = {dict(list(val.items())[:2])}")
        elif isinstance(val, list):
            print(f"  sample      : {first_key} = {val[:2]}")

# Also check hypotheses
hypos = rs.get("hypotheses", [])
print(f"\nhypotheses count: {len(hypos)}")
for h in hypos:
    print(f"  [{h.get('id')}] {h.get('statement', '')[:80]}")

# Check InsightSynthesis used dispatch citations (post-dispatch version)
after = data.get("final_insights_after") or data.get("final_insights") or {}
findings_after = after.get("key_findings", []) if isinstance(after, dict) else []
risks_after = after.get("risks_and_bias_signals", []) if isinstance(after, dict) else []
all_after = findings_after + risks_after

dispatch_cited = sum(1 for f in all_after if "dispatched[" in str(f))
print(f"\n--- Post-dispatch InsightSynthesis ---")
print(f"key_findings count  : {len(findings_after)}")
print(f"risks count         : {len(risks_after)}")
print(f"Items citing dispatched[i]: {dispatch_cited}/{len(all_after)}")
for item in all_after:
    text = str(item)
    if "dispatched[" in text:
        # Print truncated
        print(f"  > {text[:120]}")

# Compare before vs after
before = data.get("final_insights_before") or {}
findings_before = before.get("key_findings", []) if isinstance(before, dict) else []
print(f"\nkey_findings BEFORE dispatch: {len(findings_before)}")
print(f"key_findings AFTER  dispatch: {len(findings_after)}")
