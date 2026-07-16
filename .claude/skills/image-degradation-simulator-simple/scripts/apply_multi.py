#!/usr/bin/env python3
"""
Apply a sequence of x_distortion degradations to an image.

Usage:
    python apply_multi.py --input clean.png --distortions "blur_gaussian:3,noise_gaussian_RGB:2" --output degraded.png
"""

import argparse
import sys
import numpy as np
from PIL import Image

# Add project root to path so we can import x_distortion
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from x_distortion import add_distortion

def parse_distortions(dist_str):
    """Parse 'name:sev,name:sev' into list of (name, severity) tuples."""
    pairs = []
    for part in dist_str.split(","):
        part = part.strip()
        if not part:
            continue
        name, sev = part.rsplit(":", 1)
        pairs.append((name.strip(), int(sev)))
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Apply multiple x_distortion degradations in sequence")
    parser.add_argument("--input", required=True, help="Path to input (clean) image")
    parser.add_argument("--distortions", required=True,
                        help="Comma-separated name:severity pairs, e.g. 'blur_gaussian:3,noise_gaussian_RGB:2'")
    parser.add_argument("--output", required=True, help="Path to save degraded image")
    args = parser.parse_args()

    img = Image.open(args.input).convert("RGB")
    img = np.array(img, dtype=np.uint8)

    pipeline = parse_distortions(args.distortions)
    for name, severity in pipeline:
        print(f"Applying {name} (severity={severity})...")
        img = add_distortion(img, severity=severity, distortion_name=name)

    Image.fromarray(img).save(args.output)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
