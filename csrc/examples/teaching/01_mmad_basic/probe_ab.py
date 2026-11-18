#!/usr/bin/env python3
"""教学例子 01 的 A/B 口探针：同一块数据，从 A 口进和从 B 口进，读法一样吗？

方法：用单位矩阵当"透明人"（I·X = X），把同一个 arange 矩阵分别放到 A 位和 B 位。

  mmad_basic(arange, I)  ->  C = arange · Iᵀ = arange     （A 口：原样）
  mmad_basic(I, arange)  ->  C = I · arangeᵀ = arangeᵀ    （B 口：转置着理解）

两条结果应当互为转置。这就是"L0A 要 [M,K]、L0B 要 [N,K]"的行为证据，
也解释了为什么 A 用 3D 载入、B 用 2D 载入。

在 NPU 机器上跑：
    python3 csrc/examples/teaching/01_mmad_basic/probe_ab.py
"""

import torch

import torch_npu  # noqa: F401
import sgl_kernel_npu  # noqa: F401

T = 16


def main() -> int:
    ar = torch.arange(T * T, dtype=torch.float32).reshape(T, T).to(torch.float16)
    eye = torch.eye(T, dtype=torch.float16)
    ar_npu = ar.npu()
    eye_npu = eye.npu()

    c_from_a = torch.ops.npu.mmad_basic(ar_npu, eye_npu).cpu().float()  # A 口
    c_from_b = torch.ops.npu.mmad_basic(eye_npu, ar_npu).cpu().float()  # B 口

    ref = ar.float()

    print("A 口放 arange、B 位放 I：")
    print(f"  c[0,:4] = {c_from_a[0, :4].tolist()}")
    print(f"  c[1,:4] = {c_from_a[1, :4].tolist()}")
    print(f"  期望    = {ref[0, :4].tolist()} / {ref[1, :4].tolist()}   <- 原样")

    print("\nA 位放 I、B 口放 arange：")
    print(f"  c[0,:4] = {c_from_b[0, :4].tolist()}")
    print(f"  期望    = {ref.T[0, :4].tolist()}   <- 转置（0,16,32,48）")

    ok_a = torch.allclose(c_from_a, ref, atol=1e-2)
    ok_b = torch.allclose(c_from_b, ref.T, atol=1e-2)
    ok_t = torch.allclose(c_from_a, c_from_b.T, atol=1e-2)
    print(f"\nA 口输出 == arange      : {ok_a}")
    print(f"B 口输出 == arangeᵀ     : {ok_b}")
    print(f"两者互为转置            : {ok_t}")
    print("\n结论：同一个矩阵，A 口按原样喂进 cube，B 口按转置喂进 cube。")

    ok = ok_a and ok_b and ok_t
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
