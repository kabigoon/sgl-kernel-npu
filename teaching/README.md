# teaching/ — NPU 教学例子

这个目录只服务于 `learn/sparse-flash-mla` 分支：**用最小可运行的算子，把 kernel 实现的每个概念拆开**。
代码刻意写得直白（尺寸写死、不做防御性检查、不追求性能），目的是让"参数 → 实现 → 性能"的因果链一眼可见。

## 分级（按依赖顺序，一次只加一个概念）

| 级别 | 目录 | 讲什么 | 状态 |
|---|---|---|---|
| 01 | `01_mmad_basic/` | 一条 `Mmad` 的五个阶段：GM→L1→L0A/L0B→(M)→L0C→GM，16×16×16 | ✅ 已有 |
| 02 | `02_mmad_k_loop/` | K 方向分块累加：`cmatrixInitVal` / `unitFlag` / L0C 复用 | 待加 |
| 03 | `03_mmad_shapes/` | M/N 非 16 倍数、多 fractal、stride 与大 tile | 待加 |
| 04 | `04_cube_vector/` | cube 出分数块 → GM workspace → vector 做 softmax（sparse_flash_mla 的骨架） | 待加 |

## 调用流程（与 SGLang 一致）

```
python:  import sgl_kernel_npu            # dlopen libsgl_kernel_npu.so，注册 torch.ops.npu.*
         torch.ops.npu.<op>(tensors...)   # host 侧函数
host:    EXEC_KERNEL_CMD(<kernel>, blk, tensors...)   # ACLRT_LAUNCH_KERNEL
device:  extern "C" __global__ __aicore__ void <kernel>(GM_ADDR ...)   # AscendC kernel
```

这条链路和 `csrc/helloworld` 完全一样，是仓库里**最短的 Python→kernel 路径**（没有 aclnn、没有 tiling）。
真实算子 `sparse_flash_mla` 多出来的部分（opdef、tiling、aclnn、metadata）都是围绕"尺寸不能写死"展开的。

## 怎么加一个例子（三个文件 + 两行 CMake）

```
teaching/NN_xxx/
├── op_kernel/xxx_kernel.cpp   # extern "C" __global__ __aicore__ void xxx(GM_ADDR ...)
├── op_host/xxx.cpp            # EXEC_KERNEL_CMD(xxx, blockDim, ...) + TORCH_LIBRARY_FRAGMENT(np)
└── run.py                     # torch.ops.npu.xxx(...)
```

然后在 `cmake/legacy_kernels.cmake` 两处列表各加一行（host / kernel）：

```cmake
${PROJECT_SOURCE_DIR}/teaching/NN_xxx/op_host/xxx.cpp
${PROJECT_SOURCE_DIR}/teaching/NN_xxx/op_kernel/xxx_kernel.cpp
```

kernel 里的函数名 `xxx` 决定生成的启动头文件名 `aclrtlaunch_xxx.h`，三者必须一致。

## 构建 & 运行（在 NPU 机器上）

```bash
cd /home/a00821909/sgl-kernel-npu
git fetch origin && git switch learn/sparse-flash-mla && git pull

# 编译（首次 10~40 min；之后只是增量重编 teaching/ 下的文件 + 重链，通常 1~3 min）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 setup.py --arch-version 2201 --ops apply_token_bitmask -j 16

# 装 wheel（必须 no-index no-deps）
pip install --no-index --no-deps --force-reinstall output/sgl_kernel_npu-*.whl

# 跑
python3 teaching/01_mmad_basic/run.py
```

> 只关心教学例子时可以用 `--without-third-party` 跳过第三方 vendor（更快），
> 但这样打出的 wheel 里没有 `_third_party/sgl_ops_transformer`，`sparse_flash_mla` 会不可用。
> 两者别混着装。
