#!/usr/bin/env python3
"""
Qwen3.7-Plus 盲退化识别消融实验

用 Qwen 多模态模型替代 Skill Agent 进行盲识别，
验证我们 Skill-based 盲识别阶段的有效性。

用法:
  export QWEN_API_KEY="your_key"
  export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 默认
  .venv/bin/python3 exp37/scripts/qwen_blind_id.py --challenge blind_0001
  .venv/bin/python3 exp37/scripts/qwen_blind_id.py --all  # 全部 24 挑战
"""

import argparse, json, os, sys, base64, time, re
from io import BytesIO
from PIL import Image

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
# DashScope API 配置（阿里云百炼）
#   export DASHSCOPE_API_KEY="sk-xxx"
#   export DASHSCOPE_WORKSPACE_ID="your-workspace-id"  # 业务空间ID，可选
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
WORKSPACE_ID = os.environ.get("DASHSCOPE_WORKSPACE_ID", "")
if WORKSPACE_ID:
    BASE_URL = f"https://{WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
else:
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.7-plus-2026-05-26"

PHASE4 = "exp37/challenges/phase4"

# ---------------------------------------------------------------------------
# x_distortion 退化库信息（提供给 Qwen）
# ---------------------------------------------------------------------------
# 简化版退化函数列表（仅名称，severity 1-5）
FUNCTION_LIST = """
可用函数 (severity 1-5):
blur: blur_gaussian, blur_motion, blur_glass, blur_lens
noise: noise_gaussian_RGB, noise_gaussian_YCrCb, noise_speckle, noise_poisson, noise_impulse
compression: compression_jpeg
global: brightness_brighten/darken_shfit/gamma_HSV/RGB, contrast_strengthen/weaken_scale/stretch, saturate_strengthen/weaken_HSV/YCrCb, quantization_median/hist
"""

FULL_PROMPT = FUNCTION_LIST + """
约束: blur/noise/compression各最多一次, ≤3步。顺序: 噪声→模糊→全局→JPEG。

分析下方2对图像(左=原图, 右=退化)。给出退化管线。格式: "function:severity"。
必须非空。输出JSON:
{"pipeline": [...], "analysis": {"verdict": "LIKELY|UNCERTAIN|POOR", "tier": 1|2|3, "reasoning": "..."}}
"""

NOISE_GATE_PROMPT = """
分析下方2对图像(左=原图, 右=退化)。只需回答: 是否存在随机噪声(gaussian/speckle/poisson/impulse)?
回复 YES 或 NO。
"""


def encode_image(path):
    """Encode image to base64 data URL (原图 PNG, 不压缩)."""
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_prompt(challenge_dir, num_pairs=2, noise_gate=False):
    """Build prompt with image pairs. num_pairs: 2 for speed, 5 for full analysis."""
    prompt = NOISE_GATE_PROMPT if noise_gate else FULL_PROMPT
    content = [{"type": "text", "text": prompt}]

    for i in range(min(num_pairs, 5)):
        degraded_path = os.path.join(challenge_dir, f"degraded_{i}.png")
        clean_path = os.path.join(challenge_dir, f"clean_{i}.png")
        if not os.path.exists(degraded_path) or not os.path.exists(clean_path):
            continue

        degraded_b64 = encode_image(degraded_path)
        clean_b64 = encode_image(clean_path)

        content.append({"type": "text", "text": f"--- 图像对 {i+1} ---"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(clean_path)}"}})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(degraded_path)}"}})

    return content


def parse_response(text):
    """Parse Qwen response into predicted_params format."""
    # Extract JSON block
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        # Try to find raw JSON
        json_match = re.search(r'\{.*"pipeline".*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {
            "pipeline": [],
            "analysis": {
                "verdict": "POOR",
                "tier": 0,
                "reasoning": f"Failed to parse Qwen response: {text[:200]}"
            }
        }

    # Normalize pipeline format
    pipeline = data.get("pipeline", [])
    normalized = []
    for step in pipeline:
        if isinstance(step, str) and ":" in step:
            normalized.append(step)
        elif isinstance(step, dict):
            normalized.append(f"{step['function']}:{step['severity']}")

    analysis = data.get("analysis", {})
    return {
        "pipeline": normalized,
        "alternatives": data.get("alternatives", []),
        "analysis": {
            "verdict": analysis.get("verdict", "UNCERTAIN"),
            "tier": analysis.get("tier", 0),
            "reasoning": analysis.get("reasoning", ""),
            "model": MODEL,
        }
    }


def run_qwen(challenge_id, client, noise_gate=False):
    """Run Qwen blind ID on a single challenge. noise_gate=True for binary noise check."""
    challenge_dir = os.path.join(PHASE4, challenge_id)

    # Check if already done
    output_file = os.path.join(challenge_dir, "predicted_params_qwen.json" if not noise_gate else "qwen_noise_gate.json")
    if os.path.exists(output_file) and not noise_gate:
        print(f"  {challenge_id}: SKIP (already done)")
        return json.load(open(output_file))

    if not os.path.isdir(challenge_dir):
        print(f"  {challenge_id}: NO DIR")
        return None

    content = build_prompt(challenge_dir, num_pairs=2, noise_gate=noise_gate)

    print(f"  {challenge_id}: Sending {len(content)} msgs to Qwen (noise_gate={noise_gate})...")
    t0 = time.time()

    # Retry: 1 for noise gate, 5 for full blind ID
    max_retries = 1 if noise_gate else 5
    messages = [{"role": "user", "content": content}]
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=2048,
                temperature=min(0.1 + attempt * 0.2, 0.9),
            )
        except Exception as e:
            print(f"  {challenge_id}: API ERROR: {e}")
            return None

        elapsed = time.time() - t0
        text = response.choices[0].message.content

        # Noise gate: simple YES/NO parse
        if noise_gate:
            has_noise = "YES" in text.upper()
            result = {"has_noise": has_noise, "raw": text.strip(), "time_s": round(elapsed, 1)}
            print(f"  {challenge_id}: {'NOISE' if has_noise else 'CLEAN'} ({elapsed:.1f}s)")
            json.dump(result, open(output_file, "w"), indent=2)
            return result

        result = parse_response(text)
        result["analysis"]["qwen_response_time_s"] = round(elapsed, 1)
        result["analysis"]["qwen_raw_response"] = text[:500]
        result["analysis"]["qwen_attempts"] = attempt + 1

        if result["pipeline"] and len(result["pipeline"]) > 0:
            print(f"  {challenge_id}: Attempt {attempt+1}, {elapsed:.1f}s → {result['pipeline']} ({result['analysis']['verdict']})")
            json.dump(result, open(output_file, "w"), indent=2)
            return result
        else:
            print(f"  {challenge_id}: Attempt {attempt+1}, empty pipeline, retrying...")
            messages.append({"role": "assistant", "content": text})
            if attempt < 3:
                # First retries: demand non-empty
                messages.append({"role": "user", "content": "你的回复 pipeline 为空。你必须给出至少 1 步退化。即使不确定也要给出最好的猜测。重新输出完整 JSON。"})
            else:
                # Later retries: simplify task — just identify the single most obvious degradation
                messages.append({"role": "user", "content": "你仍返回了空 pipeline。现在简化任务：只识别最明显的一种退化，给出 1 步 pipeline。不要返回空数组。输出 JSON。"})

    # All retries exhausted — truly cannot identify
    result["analysis"]["verdict"] = "POOR"
    result["analysis"]["reasoning"] = f"QWEN_CANNOT_IDENTIFY: returned empty pipeline after {max_retries} attempts"
    json.dump(result, open(output_file, "w"), indent=2)
    print(f"  {challenge_id}: QWEN_CANNOT_IDENTIFY after {max_retries} attempts")
    return result


def main():
    parser = argparse.ArgumentParser(description="Qwen3.7-Plus 盲退化识别消融")
    parser.add_argument("--challenge", type=str, help="单个挑战 ID (如 blind_0001)")
    parser.add_argument("--all", action="store_true", help="全部 24 挑战")
    parser.add_argument("--start", type=int, default=1, help="起始编号 (配合 --all)")
    parser.add_argument("--end", type=int, default=24, help="结束编号 (配合 --all)")
    parser.add_argument("--noise-gate", action="store_true", help="噪声门模式: 仅判断噪声存在性 (YES/NO)")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: 请设置 DASHSCOPE_API_KEY 环境变量")
        print("  export DASHSCOPE_API_KEY='sk-xxx'")
        print("  可选: export DASHSCOPE_WORKSPACE_ID='your-workspace-id'")
        sys.exit(1)

    # Init client
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Build challenge list
    if args.all:
        challenges = [f"blind_{i:04d}" for i in range(args.start, args.end + 1)]
    elif args.challenge:
        challenges = [args.challenge]
    else:
        print("请指定 --challenge 或 --all")
        sys.exit(1)

    print(f"Qwen {'Noise Gate' if args.noise_gate else 'Blind ID'}: {len(challenges)} challenges")
    print(f"Model: {MODEL}")
    print(f"Base URL: {BASE_URL}")

    results = {}
    for bid in challenges:
        result = run_qwen(bid, client, noise_gate=args.noise_gate)
        if result:
            results[bid] = result

    # Summary
    verdicts = {}
    for bid, r in results.items():
        v = r["analysis"]["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1

    print(f"\nDone: {len(results)}/{len(challenges)}")
    print(f"Verdicts: {verdicts}")


if __name__ == "__main__":
    main()
