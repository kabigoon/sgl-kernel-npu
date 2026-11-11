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

## 一个例子的最小构成

教学例子走的是仓库**正式**的加算子通道（和 `apply_token_bitmask` 一样），因为这正是
`sparse_flash_mla` 被接进 wheel 的方式：

```
teaching/NN_xxx/
├── op_host/xxx_op.cpp        # 一个文件三个角色：opdef + infershape + tiling
├── op_host/torch_adapter.cpp # aclnn -> torch.ops.npu.xxx
├── op_kernel/xxx_kernel.cpp  # extern "C" __global__ __aicore__ void sgl_teaching(...)
└── run.py                    # torch.ops.npu.xxx(...)
```

```
python:  import sgl_kernel_npu            # dlopen libsgl_kernel_npu.so（里面有各算子的 adapter）
         torch.ops.npu.xxx(...)           # adapter
host:    aclnnSglTeachingGetWorkspaceSize -> TilingMmadBasic（算 blockDim / workspace）
         -> vendor 里的 aclnnSglTeaching
device:  sgl_teaching kernel（AscendC）
```

kernel 的入口函数名必须是 **OpType 的 snake_case**（`SglTeaching` → `sgl_teaching`），
msopgen 靠这个名字把 opdef 和 kernel 配起来；参数是 `(输入..., 输出..., workspace, tiling)`。

## 为什么所有例子共用一个算子 ID

CANN 与 torch 两级的构建缓存都是拿 `--ops` 的选择串当 key（`build_tools/cann.py:154`），
多注册一个 ID 就必须 `--make-clean` 全量重来。所以：

- 只注册一个算子：ID `teaching` / OpType `SglTeaching`
- 声明在 `csrc/examples/teaching/CMakeLists.txt`（只指路，源文件都在 `teaching/` 下）
- **换例子 = 改那 4 个源文件路径**，选择串保持 `apply_token_bitmask,teaching` 不变
- 一次只激活一个例子（同一个 ID 不能有两个实现）

## 构建 & 运行（在 NPU 机器上）

```bash
cd /home/a00821909/sgl-kernel-npu
git fetch origin && git switch learn/sparse-flash-mla && git pull
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 第一次切到新选择串（apply_token_bitmask -> apply_token_bitmask,teaching）必须清缓存
python3 setup.py --make-clean
python3 setup.py --arch-version 2201 --ops apply_token_bitmask,teaching -j 16

pip install --no-index --no-deps --force-reinstall output/sgl_kernel_npu-*.whl

python3 teaching/01_mmad_basic/run.py
```

之后再改 `teaching/` 下的文件，**不需要再 `--make-clean`**，直接重跑 `setup.py`（它内部
`cmake --build` 会按源码时间戳增量重编，vendor 包和 torch 扩展都会更新）→ 重装 wheel → 跑。

> `--make-clean` 只清 `build/cann`、`build/torch`、`build/wheel`；`build/third_party` 的
> 第三方缓存（`tp...`）会保留，所以第三方那一段是缓存命中，不是 10~40 分钟。
