#!/usr/bin/env python3
"""
Save degradation simulation results in standardized format.

Usage:
    python save_results.py \
        --output-dir degradation_results/my_test \
        --target /path/to/target.png \
        --clean /path/to/clean.png \
        --simulated /path/to/simulated.png \
        --params /path/to/params.json \
        --reflection /path/to/reflection.json \
        --iterations-dir /tmp/workspace/iterations
"""

import argparse
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description="Save degradation simulation results")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--target", required=True, help="Path to target degraded image")
    parser.add_argument("--clean", required=True, help="Path to clean source image")
    parser.add_argument("--simulated", required=True, help="Path to final simulated image")
    parser.add_argument("--params", required=True, help="Path to params.json")
    parser.add_argument("--reflection", default=None, help="Path to reflection.json")
    parser.add_argument("--iterations-dir", default=None, help="Directory containing iteration images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Copy images
    shutil.copy2(args.target, os.path.join(args.output_dir, "target.png"))
    shutil.copy2(args.clean, os.path.join(args.output_dir, "clean_source.png"))
    shutil.copy2(args.simulated, os.path.join(args.output_dir, "simulated.png"))

    # Copy params
    shutil.copy2(args.params, os.path.join(args.output_dir, "params.json"))

    # Copy reflection log
    if args.reflection and os.path.isfile(args.reflection):
        shutil.copy2(args.reflection, os.path.join(args.output_dir, "reflection.json"))

    # Copy iterations
    if args.iterations_dir and os.path.isdir(args.iterations_dir):
        iter_dest = os.path.join(args.output_dir, "iterations")
        os.makedirs(iter_dest, exist_ok=True)
        for fname in sorted(os.listdir(args.iterations_dir)):
            src = os.path.join(args.iterations_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(iter_dest, fname))

    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
