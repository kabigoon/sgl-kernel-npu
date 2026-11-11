#!/usr/bin/env python3
"""教学例子 01 的布局诊断：确认"B 的方向"这类错误。

算子契约是 C = A·Bᵀ（A=[M,K]，B=[N,K]）。这个脚本做两件事：

  1. 候选假设打分 —— 把硬件实际算出来的矩阵，和"每种可能的布局解释"比一遍，
     max|diff| ≈ 0 的那条就是当前 kernel 真实的行为。
  2. 单位矩阵探针 —— A=I 时 C 应该等于 Bᵀ；如果等于 B，说明 B 的方向又反了。

用途：以后改 per-fractal 参数（nValue/dValue/dstNzC0Stride、LoadData 的
repeatTimes/srcStride/ifTranspose、Fixpipe 的 srcStride/dstStride）之后，
先跑它确认"算的是不是我以为的那件事"，比盯着错误值猜快得多。

在 NPU 机器上跑：
    python3 csrc/examples/teaching/01_mmad_basic/diagnose.py
"""

import torch

import torch_npu  # noqa: F401
import sgl_kernel_npu  # noqa: F401

T = 16


def report(name: str, got: torch.Tensor, cand: torch.Tensor) -> float:
    diff = (got - cand).abs().max().item()
    mark = "  <== 匹配" if diff < 1e-2 else ""
    print(f"  {name:34s} max|diff| = {diff:.6e}{mark}")
    return diff


def main() -> int:
    torch.manual_seed(0)

    a = torch.randn(T, T, dtype=torch.float16, device="npu")  # A = [M, K]
    b = torch.randn(T, T, dtype=torch.float16, device="npu")  # B = [N, K]
    c = torch.ops.npu.mmad_basic(a, b).cpu().float()

    A = a.cpu().float()
    B = b.cpu().float()

    print("=== 候选假设（契约：C = A·Bᵀ）===")
    best = 1e9
    best = min(best, report("A@B.T   契约正确", c, A @ B.T))
    best = min(best, report("A@B     B 按 [K,N] 解释", c, A @ B))
    best = min(best, report("A.T@B   A 方向反了", c, A.T @ B))
    best = min(best, report("B@A.T   A/B 互换", c, B @ A.T))
    best = min(best, report("(A@B.T).T  输出方向反了", c, (A @ B.T).T))

    print("\n=== 单位矩阵探针 ===")
    print("A=I 时，契约正确的实现应输出 Bᵀ。")
    eye = torch.eye(T, dtype=torch.float16, device="npu")
    bp = torch.arange(T * T, dtype=torch.float32).reshape(T, T).to(torch.float16)
    cp = torch.ops.npu.mmad_basic(eye, bp.npu()).cpu().float()
    bpf = bp.float()
    print(f"  c    [0,:4] = {cp[0, :4].tolist()}")
    print(f"  B.T  [0,:4] = {bpf.T[0, :4].tolist()}   <- 应与此一致")
    print(f"  B    [0,:4] = {bpf[0, :4].tolist()}")

    print(f"\n最接近的假设 max|diff| = {best:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
