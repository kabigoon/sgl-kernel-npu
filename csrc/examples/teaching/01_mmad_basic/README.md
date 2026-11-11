# 01 — 一条 Mmad 到底干了什么

`C[16,16](fp32) = A[16,16](fp16) · B[16,16](fp16)ᵀ`，只用一个 AI Core、一条 Mmad。

## 文件地图

| 文件 | 角色 | 对应 sparse_flash_mla 里 |
|---|---|---|
| `op_host/mmad_basic_op.cpp` | opdef + infershape + tiling（15 行的 tiling） | `sparse_flash_mla_def.cpp` / `_infershape.cpp` / `_tiling.cpp`(2682 行) |
| `op_host/torch_adapter.cpp` | `torch.ops.npu.mmad_basic` → aclnn | 上游 Python 门面 + JIT shim |
| `op_kernel/mmad_basic_kernel.cpp` | 真正跑在 AI Core 上的 5 段流水 | `arch22/sparse_flash_mla_*_block_cube.h` |
| `op_kernel/mmad_basic_tiling.h` | host/kernel 共享的 tiling 结构（1 个字段） | `sparse_flash_mla_tiling.h`（60+ 字段） |
| `run.py` | 调用 + 对拍 | 测试用例 |
| `diagnose.py` | 布局诊断（改完分形参数先跑它） | — |

## 两个输入都是 `[行, K]` —— 这不是接口癖好

```
A: [M, K]      L0A 的分形 zZ，K 在最里层（C0 = 16）
B: [N, K]      L0B 的分形 nZ，K 也在最里层
C: [M, N]      L0C，fp32
```

一次 Mmad 算的是 `A · Bᵀ`。因为 cube 的 L0A/L0B 分形里 **K 永远是里层（C0=16）**，
所以第二块操作数天然是"转置着放"的。

这不是怪癖，是 attention 需要的形状：`Q[M, D] · KV[S2, D]ᵀ` —— Q 和 KV 的 D 都在里层，
一次 Mmad 直接得到分数，不需要任何转置。

> 我们第一版就是把 `B` 按数学课本的 `[K, N]` 摆的，结果算出来是 `A·B`（等价于 B 被转置）。
> `diagnose.py` 就是为这种错准备的：它会把"硬件实际算的是哪一个"指出来。

## 为什么是 16×16×16

- fp16 的 `C0 = 16`，所以一个 fractal = `16 × 16` 个元素 = 512 B
- L0A / L0B / L0C 都按 512 B 粒度组织
- 取 M=N=K=16，正好每个 buffer 只装**一块**，没有任何循环

于是代码的每个函数都对应硬件流水的一段，且各只做一次：

| 函数 | 流水 | 干什么 | 本例搬运量 |
|---|---|---|---|
| `CopyInGmToL1` | MTE2 | GM(ND) → L1(nZ)，`Nd2NzParams` 顺手重排 | A 512 B + B 512 B |
| `LoadL1ToL0` | MTE1 | L1 → L0A(zZ，3D 载入) / L0B(nZ，2D 载入) | 512 B + 512 B |
| `MmadOneTile` | M (cube) | L0A × L0B → L0C，fp32 | — |
| `CopyOutL0cToGm` | FIX | `Fixpipe` 把 L0C 的 fp32 分形变回 ND | 1024 B |

## 三个要盯住的参数语义

| 参数 | 本例值 | 含义 | 真实 kernel 里怎么变 |
|---|---|---|---|
| `MmadParams::m/n/k` | 16/16/16 | **这一次**乘加的形状 | sparse_flash_mla 是 `128/128/128`（`swa_block_cube.h:60,61,351,728-730`） |
| `cmatrixInitVal` | `true` | `C = A·Bᵀ`（覆盖 L0C）；`false` 才是 `C += A·Bᵀ` | K 方向第一块 `true`，之后全部 `false` |
| `cmatrixSource` | `false` | 初值取自 L0C 自身 | 同理 |
| `unitFlag` | `0b11` | "我这条是最后一条，Fixpipe 可以搬了" | K 循环中间 `0b10`，最后一条 `0b11` |

## 预期输出

```
a: (16, 16) torch.float16   b: (16, 16) torch.float16   (B 是 [N,K])
c: (16, 16) torch.float32   C = A · Bᵀ   (cube 内部按 fp32 累加)
max |c - ref| = 1.xxx e-06
c[0,0]   = ...
手算 A[0,:]·B[0,:] = ...
一条 Mmad = 16x16x16 = 4096 次乘加 = 8192 FLOP
一个 fp16 fractal = 16x16 = 512 B；本例每个阶段只搬 1 块
PASS
```

`max |c - ref|` 应该是 1e-6 量级：输入是 fp16，但乘加在 fp32 里做，参考结果在 CPU 上也是 fp32。

## 编译 & 运行

```bash
cd /home/a00821909/sgl-kernel-npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3 setup.py --make-clean                                  # 仅首次：选择串变了
python3 setup.py --arch-version 2201 --ops apply_token_bitmask,teaching -j 16
pip install --no-index --no-deps --force-reinstall output/sgl_kernel_npu-*.whl
python3 csrc/examples/teaching/01_mmad_basic/run.py
```

之后改例子只需重跑后三条（不要 `--make-clean`）。改过 per-fractal 参数的话，先跑
`diagnose.py` 确认"算的还是我以为的那件事"。

## 踩过的坑（按发生顺序）

1. **wheel 里没有算子** → `setup.py` 的 torch 扩展只编被 `--ops` 选中算子的 `TORCH_ADAPTER`，`cmake/legacy_kernels.cmake` 是老 `build.sh` 的通道，不参与
2. **`Unsafe relative path: ../teaching/...`** → CANN 的 kernel 源码拷贝器要求 kernel 位于 `csrc` 子树内（`csrc/cmake/cann_ops.cmake:198` 以 `csrc` 为基准算相对路径）
3. **`gert::Shape has no member GetStorageShape`** → `InferShapeContext::GetInputShape` 返回的就是 `gert::Shape*`；`TilingContext` 那边才是 `StorageShape*`
4. **结果等于 `A·Bᵀ`** → B 必须按 `[N, K]` 摆（本文件第二节）
