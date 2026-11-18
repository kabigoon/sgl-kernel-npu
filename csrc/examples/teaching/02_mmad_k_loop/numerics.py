#!/usr/bin/env python3
"""教学例子 02 配套：为什么累加要用 fp32（需要 numpy）。

只跑 CPU，不需要 NPU。用 numpy 的 `cumsum(dtype=...)` **顺序**累加来模拟硬件累加器
（numpy 的 `sum()` 会用高精度求和，不能拿来模拟）。

    python3 csrc/examples/teaching/02_mmad_k_loop/numerics.py

关键结论：
  * fp16 的相对精度固定 ~1e-3，但**绝对**精度随累加器变大而变粗
  * attention 的 softmax 分母 Z 正是"几百项加起来接近几百"，尖锐 softmax 下
    fp16 累加会把小项整体吞掉，误差可达 5%
  * fp32 累加把同一场景的误差压到 1e-7 量级
"""

import numpy as np


def sequential_sum(values, dtype):
    """顺序累加：每一步都按 dtype 舍入，等价于硬件累加器。"""
    return np.cumsum(values.astype(dtype), dtype=dtype)[-1]


def section(title):
    print(f"\n=== {title} ===")


def main() -> int:
    section("1. fp16 / fp32 的绝对分辨率（np.spacing = 相邻可表示数的间隔）")
    mags = np.array([1, 16, 64, 256, 512], dtype=np.float32)
    print("  累加器量级 |    fp16 间隔 |   fp32 间隔")
    for m in mags:
        print(f"  {int(m):>10} | {float(np.spacing(np.float16(m))):>12.6g} | {float(np.spacing(np.float32(m))):>11.3g}")
    print("  → fp16 的间隔跟着量级变粗：到 256 时，小于 0.125 的项加进去等于没加")

    section("2. softmax 分母 Z（512 项）在两种分布下的累加误差")
    rng = np.random.default_rng(0)

    flat = np.exp(rng.normal(0, 0.5, 512))
    flat = flat / flat.max()
    sharp = np.concatenate([[1.0], np.full(511, 1e-4)])

    for name, ex in (("平坦 softmax", flat), ("尖锐 softmax", sharp)):
        ref = ex.sum(dtype=np.float64)
        z32 = float(sequential_sum(ex, np.float32))
        z16 = float(sequential_sum(ex, np.float16))
        print(f"  [{name}] 参考 Z = {ref:.6f}")
        print(f"      fp32 累加 = {z32:.6f}   相对误差 {abs(z32 - ref) / ref:.2e}")
        print(f"      fp16 累加 = {z16:.6f}   相对误差 {abs(z16 - ref) / ref:.2e}")

    section("3. 一次加法就能把小数吞掉（fp16）")
    for big, small in ((200.0, 0.05), (100.0, 0.01), (1.0, 1e-4)):
        got = float(np.float16(np.float16(big) + np.float16(small)))
        print(f"  {big:>7} + {small:<7} = {got:<10} (真值 {big + small})")

    section("4. 例子 02 实测的 3e-5 与 fp32 ULP 对得上吗")
    c = np.float32(22.6)  # C[0,0] 量级 ≈ sqrt(512)
    ulp = float(np.spacing(c))
    print(f"  |C|≈22.6 时 fp32 间隔 = {ulp:.3e}")
    print(f"  512 次累加、随机游走估计 = {ulp * np.sqrt(512):.3e}   (实测 K=512 时 3.05e-05)")
    print("  → 误差来自 fp32 累加顺序，不是 kernel 写错")

    print("\n一句话：fp16 累加的问题不是“每次都差千分之一”，而是“小项被整体吞掉”；")
    print("        attention 的 Z 正好踩在这个坑上，所以 L0C 必须是 fp32。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
