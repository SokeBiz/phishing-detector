#!/usr/bin/env python3
"""CLI entry point for Phishing Email Detector."""

import argparse
import json
import sys
import textwrap

from detector.predict import predict, predict_from_file


def format_result(result: dict) -> str:
    """Format prediction result for display."""
    lines = [
        "═" * 60,
        f"  Verdict: {result['verdict']}",
        f"  Confidence: {result['confidence']:.2%}",
        f"  Phishing Probability: {result['phishing_probability']:.2%}",
        "═" * 60,
        "\nTop Risk Indicators:",
    ]
    for name, importance in result.get("top_risk_factors", [])[:5]:
        lines.append(f"  • {name}: {importance:.4f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Phishing Email Detector — ML-based email threat analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py -s "URGENT" -b "Click here to verify: http://evil.com"
              python main.py -f email.txt
              python main.py -s "Meeting tomorrow" -b "Let's sync at 3pm" --json
        """),
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-s", "--subject", help="Email subject line")
    input_group.add_argument("-f", "--file", help="Path to email text file")

    parser.add_argument("-b", "--body", default="", help="Email body text")
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )
    parser.add_argument(
        "--model-dir", default=None, help="Custom model directory"
    )

    args = parser.parse_args()

    try:
        if args.file:
            kwargs = {"model_dir": args.model_dir} if args.model_dir else {}
            result = predict_from_file(args.file, **kwargs)
        else:
            kwargs = {"model_dir": args.model_dir} if args.model_dir else {}
            result = predict(args.subject, args.body, **kwargs)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Run model training first: python3 -m src.detector.model", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing email: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_result(result))

    sys.exit(0 if result["verdict"] == "LEGITIMATE" else 1)


if __name__ == "__main__":
    main()
