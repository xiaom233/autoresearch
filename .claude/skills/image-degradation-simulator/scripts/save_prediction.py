#!/usr/bin/env python3
"""Save predicted degradation params with enforced consistent format."""
import json, sys, os

def main():
    if len(sys.argv) < 3:
        print("Usage: save_prediction.py <output.json> <function1:sev1> [function2:sev2] ... [--verdict GOOD|NEEDS_WORK|POOR] [--ci N/10]")
        sys.exit(1)

    output = sys.argv[1]
    args = sys.argv[2:]

    pipeline = []
    verdict = "NEEDS_WORK"
    ci = "0/10"

    for arg in args:
        if arg.startswith("--verdict"):
            verdict = args[args.index(arg) + 1]
        elif arg.startswith("--ci"):
            ci = args[args.index(arg) + 1]
        elif ":" in arg and not arg.startswith("--"):
            func, sev = arg.split(":")
            pipeline.append({"function": func, "severity": int(sev)})

    result = {
        "pipeline": pipeline,
        "analysis": {
            "verdict": verdict,
            "ci_pass_rate": ci
        }
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, 'w') as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"Saved: {output}")

if __name__ == "__main__":
    main()
