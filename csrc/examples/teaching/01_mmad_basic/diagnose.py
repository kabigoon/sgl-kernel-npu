#!/usr/bin/env python3
"""教学例子 01 的诊断脚本：run.py 出了错值，先定位"错在哪一层"。

判据只有两条：
  1. 排序后元素集合是否相同 —— 相同说明只是"位置排错了"（布局 bug），
     不同说明数值本身就不对（累加/清零/精度 bug）。
  2. 拿若干候选矩阵乘去比对 max|diff|，≈0 的那个就是硬件实际做的事。

在 NPU 机器上跑：
    python3 csrc/examples/teaching/01_mmad_basic/diagnose.py
"""

import torch

import torch_npu  # noqa: F401
import sgl_kernel_npu  # noqa: F401

TORCH_T = 16


def report(name: str, got: torch.Tensor, cand: torch.Tensor) -> float:
    diff = (got - cand).abs().max().item()
    mark = "  <== 匹配" if diff < 1e-2 else ""
    print(f"  {name:34s} max|diff| = {diff:.6e}{mark}")
    return diff


def main() -> int:
    torch.manual_seed(0)

    a = torch.randn(TORCH_T, TORCH_T, dtype=torch.float16, device="npu")
    b = torch.randn(TORCH_T, TORCH_T, dtype=torch.float16, device="npu")
    c = torch.ops.npu.mmad_basic(a, b).cpu().float()

    A = a.cpu().float()
    B = b.cpu().float()
    ref = A @ B

    print("=== 1. 是纯排列错误吗 ===")
    same_multiset = torch.allclose(c.flatten().sort().values, ref.flatten().sort().values, atol=1e-2)
    print(f"排序后元素集合相同: {same_multiset}  "
          f"({'只是位置排错 -> 布局 bug' if same_multiset else '数值本身不对 -> 累加/清零/精度'})")

    print("\n=== 2. 候选假设 ===")
    t = 1e9
    t = min(t, report("A@B        (正确)", c, ref))
    t = min(t, report("A@B.T      (B 被转置)", c, A @ B.T))
    t = min(t, report("A.T@B      (A 被转置)", c, A.T @ B))
    t = min(t, report("B@A        (A/B 互换)", c, B @ A))
    t = min(t, report("(A@B).T    (输出被转置)", c, ref.T))
    t = min(t, report("A.T@B.T    (两边都转)", c, A.T @ B.T))

    print("\n=== 3. 单位矩阵探针 ===")
    print("A=I 时，正确实现应输出 B 本身；若输出 B.T，说明 B 的分形方向反了。")
    eye = torch.eye(TORCH_T, dtype=torch.float16, device="npu")
    bp = torch.arange(TORCH_T * TORCH_T, dtype=torch.float32).reshape(TORCH_T, TORCH_T)
    bp = bp.to(torch.float16)
    cp = torch.ops.npu.mmad_basic(eye, bp.npu()).cpu().float()
    bpf = bp.float()
    print(f"  c   [0,:4] = {cp[0, :4].tolist()}")
    print(f"  B   [0,:4] = {bpf[0, :4].tolist()}   <- 正确实现应与此一致")
    print(f"  B.T [0,:4] = {bpf.T[0, :4].tolist()}")

    print(f"\n最接近的假设 max|diff| = {t:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
