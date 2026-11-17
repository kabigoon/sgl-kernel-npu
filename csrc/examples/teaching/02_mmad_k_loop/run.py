#!/usr/bin/env python3
"""教学例子 02：K 方向分块累加。

契约和例子 01 一样：C[M,N] = A[M,K] · B[N,K]ᵀ（A 是 [M,K]，B 是 [N,K]）。
变化的是 K：kernel 沿 K 每 16 个元素做一条 Mmad，累加进同一个 L0C。

K = 16 时 kLoop = 1，行为与例子 01 完全相同 —— 02 是 01 的超集。

在 NPU 机器上跑：
    python3 csrc/examples/teaching/02_mmad_k_loop/run.py
"""

import torch

import torch_npu  # noqa: F401  # 让 npu 后端可用
import sgl_kernel_npu  # noqa: F401  # 注册 torch.ops.npu.*

M, N = 16, 16


def check(k: int) -> bool:
    """对给定的 K 跑一次，和 CPU 上的 fp32 参考实现对比。"""
    a = torch.randn(M, k, dtype=torch.float16, device="npu")
    b = torch.randn(N, k, dtype=torch.float16, device="npu")

    c = torch.ops.npu.mmad_basic(a, b)

    ref = a.cpu().float() @ b.cpu().float().T
    diff = (c.cpu() - ref).abs().max().item()
    ok = torch.allclose(c.cpu(), ref, atol=1e-3, rtol=1e-3)

    note = "  (= 例子 01)" if k == 16 else ""
    print(f"K={k:4d}  kLoop={k // 16:2d}  Mmad条数={k // 16:2d}  "
          f"max|c-ref|={diff:.3e}  {'PASS' if ok else 'FAIL'}{note}")
    return ok


def main() -> int:
    torch.manual_seed(0)

    print(f"A: [{M}, K] fp16   B: [{N}, K] fp16   C: [{M}, {N}] fp32   (B 是 [N,K])")
    print("同一份 kernel，不同 K，只靠 tiling data 里的 kLoop / kTotal 驱动\n")

    results = [check(k) for k in (16, 64, 128, 512)]

    print()
    print("要看的三件事：")
    print("  1. K 越大，Mmad 条数越多（每条吃 16 个 K 元素），但 C 的形状不变")
    print("  2. 每条 Mmad 都往同一个 L0C 累加：第 1 条清零，其余 cmatrixInitVal=false")
    print("  3. 只有最后一条 unitFlag=0b11，所以 Fixpipe 只搬一次（循环结束后）")

    ok = all(results)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
