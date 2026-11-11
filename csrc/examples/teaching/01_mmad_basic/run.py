#!/usr/bin/env python3
"""教学例子 01：一条 Mmad 到底干了什么。

调用流程和仓库里其它算子（含 sparse_flash_mla）完全一致：

    import sgl_kernel_npu                 -> dlopen libsgl_kernel_npu.so
    torch.ops.npu.mmad_basic(a, b)        -> torch_adapter.cpp
                                          -> aclnnSglTeaching（vendor 里的算子包）
                                          -> host tiling：blockDim = 1
                                          -> 设备侧 sgl_teaching kernel
                                          -> 一条 16x16x16 的 Mmad

算子契约：C[M,N] = A[M,K] · B[N,K]ᵀ
  A 和 B 都是 [行, K] 布局，这是 cube 的硬约束（L0A/L0B 的分形里 K 在最里层），
  也正好是 attention 里 Q[M,D] · KV[S2,D]ᵀ 的形状。

在 NPU 机器上跑：
    python3 csrc/examples/teaching/01_mmad_basic/run.py
"""

import torch

import torch_npu  # noqa: F401  # 让 npu 后端可用
import sgl_kernel_npu  # noqa: F401  # 注册 torch.ops.npu.*

M, N, K = 16, 16, 16


def main() -> int:
    torch.manual_seed(0)

    # A: [M, K]，B: [N, K]（注意不是 [K, N]）
    a = torch.randn(M, K, dtype=torch.float16, device="npu")
    b = torch.randn(N, K, dtype=torch.float16, device="npu")

    c = torch.ops.npu.mmad_basic(a, b)

    print(f"a: {tuple(a.shape)} {a.dtype}   b: {tuple(b.shape)} {b.dtype}   (B 是 [N,K])")
    print(f"c: {tuple(c.shape)} {c.dtype}   C = A · Bᵀ   (cube 内部按 fp32 累加)")

    # 参考结果放到 CPU 上用 fp32 算，避开 NPU 上的任何 cube 路径
    ref = a.cpu().float() @ b.cpu().float().T
    diff = (c.cpu() - ref).abs()
    print(f"max |c - ref| = {diff.max().item():.3e}")

    # 抽出 (0,0) 手算一遍：C[0,0] = A 的第 0 行 · B 的第 0 行
    manual = (a[0].cpu().float() * b[0].cpu().float()).sum()
    print(f"c[0,0]   = {c[0, 0].item():.6f}")
    print(f"手算 A[0,:]·B[0,:] = {manual.item():.6f}")

    print(f"一条 Mmad = {M}x{N}x{K} = {M * N * K} 次乘加 = {2 * M * N * K} FLOP")
    print(f"一个 fp16 fractal = 16x16 = {16 * 16 * 2} B；本例每个阶段只搬 1 块")

    ok = torch.allclose(c.cpu(), ref, atol=1e-3, rtol=1e-3)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
