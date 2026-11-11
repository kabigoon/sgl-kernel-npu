# 01 — 一条 Mmad 到底干了什么

`C[16,16](fp32) = A[16,16](fp16) @ B[16,16](fp16)`，只用一个 AI Core、一条 Mmad。

## 文件地图

| 文件 | 角色 | 对应 sparse_flash_mla 里 |
|---|---|---|
| `op_host/mmad_basic_op.cpp` | opdef + infershape + tiling（15 行的 tiling） | `sparse_flash_mla_def.cpp` / `_infershape.cpp` / `_tiling.cpp`(2682 行) |
| `op_host/torch_adapter.cpp` | `torch.ops.npu.mmad_basic` → aclnn | 上游 Python 门面 + JIT shim |
| `op_kernel/mmad_basic_kernel.cpp` | 真正跑在 AI Core 上的 5 段流水 | `arch22/sparse_flash_mla_*_block_cube.h` |
| `op_kernel/mmad_basic_tiling.h` | host/kernel 共享的 tiling 结构（1 个字段） | `sparse_flash_mla_tiling.h`（60+ 字段） |

## 为什么是 16×16×16

Ascend 的 cube 不是"任意矩阵乘"，它的最小工作单元是 **fractal**：

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

`Mmad` 只认 L0A/L0B 里的 fractal，**不认 GM 里的 ND**；`Fixpipe` 也只从 L0C 搬。
这就是"手写 Mmad"比 `Matmul` 高层接口多出来的全部工作量。

## 三个要盯住的参数语义

| 参数 | 本例值 | 含义 | 真实 kernel 里怎么变 |
|---|---|---|---|
| `MmadParams::m/n/k` | 16/16/16 | **这一次**乘加的形状 | sparse_flash_mla 是 `128/128/128`（`swa_block_cube.h:60,61,351,728-730`） |
| `cmatrixInitVal` | `true` | `C = A*B`（覆盖 L0C）；`false` 才是 `C += A*B` | K 方向第一块 `true`，之后全部 `false` |
| `cmatrixSource` | `false` | 初值取自 L0C 自身 | 同理 |
| `unitFlag` | `0b11` | "我这条是最后一条，Fixpipe 可以搬了" | K 循环中间 `0b10`，最后一条 `0b11` |

`cmatrixInitVal` 那条就是 attention 里 `S = Q·Kᵀ` 能做 K 分块累加的原因：
第 2 个 K block 不清零，直接往上加。

## 预期输出

```
a: (16, 16) torch.float16   b: (16, 16) torch.float16
c: (16, 16) torch.float32   (cube 内部按 fp32 累加)
max |c - ref| = 2.xxx e-06
c[0,0]   = ...
手算 A[0,:]·B[:,0] = ...
一条 Mmad = 16x16x16 = 4096 次乘加 = 8192 FLOP
一个 fp16 fractal = 16x16 = 512 B；本例每个阶段只搬 1 块
PASS
```

`max |c - ref|` 应该是 1e-6 量级：输入是 fp16，但乘加在 fp32 里做，
参考结果在 CPU 上也是 fp32，差的只是累加顺序。

## 编译 & 运行

```bash
cd /home/a00821909/sgl-kernel-npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3 setup.py --make-clean                                  # 仅首次：选择串变了
python3 setup.py --arch-version 2201 --ops apply_token_bitmask,teaching -j 16
pip install --no-index --no-deps --force-reinstall output/sgl_kernel_npu-*.whl
python3 teaching/01_mmad_basic/run.py
```

之后改例子只需重跑后三条（不要 `--make-clean`）。

## 如果编不过，大概率是这几处

1. `aclnn_sgl_teaching.h: No such file` → vendor 没装到 `SGL_CANN_VENDOR/op_api/include`，
   或者 CANN 侧 codegen 没跑到（`build/cann/kernel-build.log`）
2. `sgl_teaching` 未定义 / kernel 没编进去 → 入口函数名与 OpType 的 snake_case 不一致
3. 输出全 0 或乱码 → `FixpipeParamsV220` 的 `srcStride/dstStride/unitFlag` 没配对
4. `Tiling data size exceeds capacity` → `mmad_basic_tiling.h` 里加了字段但没同步
   `SetDataSize`（本例只有 4 B，一般不会）
