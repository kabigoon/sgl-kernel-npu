# Sparse Flash MLA 学习笔记 & 会话交接指南

> **这份文档是什么**：把「把 ops-transformer 的 `sparse_flash_mla` 迁移进 sgl-kernel-npu」这件事，
> 从任务背景、环境、构建、代码结构、数学语义、已踩的坑，到**学习进度**全部记录下来。
>
> **给谁看**：接替上一个会话继续讲算子实现的新会话（AI），以及任何想快速上手这个算子的同事。
>
> **怎么用**：新会话开工前先完整读一遍；讲课时以第 5、6、7 节的模型为骨架，以第 4 节的代码为落点。
>
> **维护**：每讲完一块，更新第 9 节「学习进度」。

---

## 0. 30 秒速览

- **任务**：把 CANN `ops-transformer` 仓库里的 `sparse_flash_mla` / `sparse_flash_mla_metadata` 接入本仓库，目标平台 **A3（arch22）**。算子本体留在上游，本仓库只做适配 + 打包 + 测试 + 文档。
- **状态**：代码、打包、文档、测试已完成，**A3 上 6 个用例全部通过**（含与独立 Python 参考实现的精度对拍）。
- **当前阶段**：从"会用/懂原理"转向"读懂 kernel 实现"。概念层已经讲完（见第 6、7 节），待讲的是 `op_host/tiling` 与 `op_kernel` 的具体实现。
- **待办**：性能数据（需 NPU 空闲）、PR 描述、commit pin 讨论（见第 9.3 节）。

---

## 1. 任务背景与当前状态

### 1.1 任务定义

| 项 | 内容 |
|---|---|
| 迁移对象 | `attention/sparse_flash_mla`（主算子）+ `sparse_flash_mla_metadata`（调度表算子） |
| 上游仓库 | `https://gitcode.com/cann/ops-transformer` |
| 上游 commit | `7df757adc542c75733f9702af97ee7a14fca9f51`（会议指定；后续 review 可能改） |
| 目标平台 | A3 / `ascend910_93` / `arch22` |
| 交付形态 | `sgl-kernel-npu` wheel 内置 ops-transformer vendor + Python 绑定 |

### 1.2 分支与提交

- 分支：`feat/sparse-flash-mla`（**不要再往这个分支加学习文档**）
- 基线：`upstream/main`（其中已包含已合入的 PR #940 `14e8c39`）
- 本 PR 的提交（自下而上）：

| commit | 内容 |
|---|---|
| `c7c145f` | `[feat] add sparse_flash_mla operators from ops-transformer`（主体） |
| `1d54321` | 去掉独立 `cann_ops_transformer` 兜底导入；收紧测试 |
| `eaff062` | 去掉测试里冗余的输出断言 |

- 变更规模：11 个文件，+1494/−2
- 学习分支：`learn/sparse-flash-mla`（本分支，只加这份文档）

### 1.3 改动清单

| 文件 | 作用 | 进 wheel |
|---|---|---|
| `python/sgl_kernel_npu/sgl_kernel_npu/operators/attention/sparse_flash_mla.py` | **包装层**：注册、CANN 版本门禁、转发 | ✅ |
| `.../implementation/torch_native/attention/sparse_flash_mla.py` | **独立 FP32 参考实现**（对拍用） | ✅ |
| `.../operators/attention/__init__.py` | 导出两个公开函数（+3 行） | ✅ |
| `tests/python/operators/attention/test_sparse_flash_mla.py` | 6 个用例 | ❌ |
| `third_party/profiles/cann-9.2.0/ops_transformer.json` | 第三方构建配置（arch22 加两个 op；commit pin） | ❌（构建期） |
| `docs/api/{zh,en}/operators/attention/sparse_flash_mla{,_metadata}.md` | 双语文档 | ❌ |
| `docs/api/{zh,en}/API_list.md` | 索引（+2 行） | ❌ |

---

## 2. 环境与仓库

### 2.1 两台机器

| 角色 | 位置 | 说明 |
|---|---|---|
| 开发机（Mac） | 本仓库 checkout：`/Users/an/Developer/sgl-kernel-npu` | **没有 NPU**，只能做静态检查、跑参考实现逻辑 |
| NPU 机（A3） | `/home/a00821909/sgl-kernel-npu`，root 用户 | 真正跑测试/性能的地方；**由用户手动执行命令并把输出贴回来** |
| 上游参考代码 | `/Users/an/Developer/ops-transformer`（Mac） | `master` = `23b4626b`；**pin 是 `7df757ad`**；该目录在沙箱工作区之外，`git checkout` 需要额外授权 |

### 2.2 NPU 机关键版本

```
CANN             9.2.0          （auto 门禁硬要求）
torch            2.13.0+cpu
torch_npu        2.13.0
Python           3.12.15
site-packages    /usr/local/python3.12.15/lib/python3.12/site-packages/sgl_kernel_npu
芯片             ascend910_93（A3）
```

### 2.3 沙箱与写入限制（给 AI 会话）

- 工作区：`/Users/an/Developer/sgl-kernel-npu`（可自由读写）
- `/tmp` 可写但**重启会被清空**（之前的 ops-transformer 克隆就是这样丢的）
- 写 `/Users/an/Developer` 之外需要 `danger-full-access` 授权（一次性）
- 因此：**不要把重要材料放在 `/tmp`**

---

## 3. 构建、安装与运行

### 3.1 六个阶段

```
① resolve_plan        读 profile → 决定 commit / ops / soc
② build_dependencies  git clone 上游@commit → build.sh 两次 → 装 .run → 校验
③ build_cann          编本仓库 csrc 里的 AscendC 算子
④ build_torch         编 libsgl_kernel_npu.so，并把上面所有产物 stage 进来
⑤ build_wheel         打成 sgl_kernel_npu-*.whl
⑥ pip install         解压到 site-packages
```

### 3.2 上游产出两个东西（容易记混）

| 命令 | 产物 | 去向 |
|---|---|---|
| `build.sh --pkg` | `build_out/*.run`（**安装器**，不是 wheel） | 装到临时目录 → vendor 目录拷进 `_cann/vendors/` |
| `build.sh --torch_extension` | `torch_extension/dist/*.whl` | 抽 Python 包 + 改写 import → `_third_party/sgl_ops_transformer/` |

### 3.3 命令

```bash
# NPU 机上
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 只看构建计划（不编译）
python3 setup.py --arch-version 2201 --plan

# 完整构建（含第三方，10~40 分钟）
python3 setup.py --arch-version 2201 --ops apply_token_bitmask -j 16

# 只编本仓库、跳过第三方
python3 setup.py --arch-version 2201 --ops apply_token_bitmask --without-third-party

# 改过配置后清理
python3 setup.py --make_clean

# 安装（必须 no-index no-deps，否则会去 PyPI 拉 427MB 的 torch）
pip install --no-index --no-deps --force-reinstall output/sgl_kernel_npu-*.whl

# 跑测试
python3 -m pytest tests/python/operators/attention/test_sparse_flash_mla.py \
  --skip-acc-check=false --enable-perf=false -v -s -rs
```

### 3.4 缓存

| 缓存 | 位置 | 失效条件 |
|---|---|---|
| 第三方构建 | `build/third_party/cache/<指纹>.json` | profile/commit/ops/soc/CANN 版本/命令变化 |
| 本仓库 CANN | `build/cann/` | 算子选择/arch 变化 |
| JIT host 扩展 | `~/.cache/torch_extensions/sparse_flash_mla/` + `.compile.fprint` | csrc/torch 头文件变化 |
| ccache | 系统级 | —— |

### 3.5 wheel 文件名解码

```
sgl_kernel_npu-2026.9.0-1refactor2201custom<b8bd28f876>tp<cb56408362>-cp312-cp312-linux_aarch64.whl
                      └体系┘└arch┘└选择┘└算子指纹┘ └第三方指纹┘ └── ABI tag ──┘
```

出现 `tp...` 就说明**第三方依赖已打进 wheel**。

---

## 4. 代码结构（当前 HEAD 实测）

### 4.1 包装层 `operators/attention/sparse_flash_mla.py`（270 行）

| 行 | 内容 |
|---|---|
| 3-8 | `detect_cann_version`、`Generation`、registry；`_CANN_FNS` / `_NATIVE_FN` 缓存 |
| 11-27 | `_load_cann_fns()`：**只允许**从 `_third_party.sgl_ops_transformer` 导入；缺依赖时抛 `RuntimeError` 提示"用第三方 profile 构建" |
| 29-43 | `_auto_impl`：检查 CANN == 9.2.0 → 转发给 `cann_ops_transformers_cann920` |
| 45-52 | `_cann_impl`：调 `_load_cann_fns()[0]`（上游主算子） |
| 54-63 | `_native_impl`：调本地参考实现（缓存到 `_NATIVE_FN`） |
| 66-171 | 公开接口 `sparse_flash_mla`：28 个参数 + docstring + 显式转发 |
| 173-197 | metadata 的 auto / cann 两个 impl |
| 199-270 | 公开接口 `sparse_flash_mla_metadata` |

### 4.2 参考实现 `implementation/torch_native/attention/sparse_flash_mla.py`（310 行）

| 行 | 函数 | 作用 |
|---|---|---|
| 86 | `_ori_window` | 算 `[s_i, e_i)`（含 θ 偏移、窗口、截断） |
| 100 | `_cmp_limit` | 算 κ（压缩侧因果上界，含 floor、残差） |
| 108 | `_cmp_sparse_keys` | CSA 索引过滤（`-1` 终止、越界跳过、topk 上限） |
| 121 | `_softmax_with_sinks` | FP32 softmax + sink 进分母 + LSE |
| 134 | `sparse_flash_mla` | 主流程：拆分 batch → 逐 head/逐行 → 拼接 key → 打分 → 输出 |

### 4.3 测试 `tests/python/operators/attention/test_sparse_flash_mla.py`（221 行）

- 第 1-15 行：**模块级先 `import sgl_kernel_npu`**（关键！见 8.2 坑 4）
- 6 个用例：`swa`、`csa-ratio-2`、`csa-ratio-4`、`hca-ratio-128`、`decode-csa`（16 请求 × 8 token，ori 16008，ratio 4）、`bsnd_swa`
- 断言：metadata `int32` 一维；输出 shape/dtype/device；LSE 形状；`check_accuracy(auto vs torch_native)`；`check_performance`

### 4.4 profile `third_party/profiles/cann-9.2.0/ops_transformer.json`

```json
"ops": {
  "arch22": ["sparse_flash_mla", "sparse_flash_mla_metadata"],
  "arch35": ["flash_mla", "flash_mla_metadata"]
},
"commit": "7df757adc542c75733f9702af97ee7a14fca9f51"
```

- `commit` 是**仓库级**的：改它会连带改变 arch35 的 `flash_mla` 源码版本
- `has_aicpu: true`：metadata 算子是 AICPU kernel

---

## 5. 调用链（从 pytest 到 kernel）

```
test_*.py
 └─ sparse_flash_mla(q, ...)                     Python 门面（默认 impl="auto"）
     └─ get_impl_by_name("...", "auto")           Python 注册表（dict）
         └─ _auto_impl                            CANN 版本门禁
             └─ get_impl_by_name("...", "cann_ops_transformers_cann920")
                 └─ _cann_impl
                     └─ _load_cann_fns()[0]      _third_party 里的上游 Python 函数
                         └─ OpBuilder.load()     首次 JIT 编 C++ shim（~30s，缓存后秒开）
                             └─ aclnnSparseFlashMla   来自 vendor 的 libcust_opapi.so
                                 └─ host tiling      算 tiling key + tiling data
                                     └─ runtime 按 key 选 kernel 实例
                                         └─ AICore kernel（.o） + AICPU metadata
```

关键点：
- **Python 层只负责"选哪个实现 + 转发"**，不参与计算
- 上游 Python 函数是"第二层门面"，也是薄薄一层
- 真正的开关在 **tiling key**（第 7.4 节）

---

## 6. 算子数学语义

### 6.1 公式

对第 $i$ 个 query（batch 内 0-based）、第 $n$ 个 head：

$$
S_i=\underbrace{[\max(j_i-127,0),\ j_i+1)}_{\text{ori 窗口}}\ \cup\ \underbrace{\mathcal{C}_i}_{\text{压缩选中}},
\qquad j_i = i + (L_{kv}-L_q)
$$

$$
m_i=\max\Big(\max_{j\in S_i}\alpha q_i\cdot k_j,\ \sigma_n\Big),\quad
O_i=\frac{\sum_{j\in S_i}e^{\alpha q_i\cdot k_j-m_i}k_j}{\sum_{j\in S_i}e^{\alpha q_i\cdot k_j-m_i}+e^{\sigma_n-m_i}}
$$

- $V=K$（共享 KV，MLA 特征）
- $\alpha$ = `softmax_scale`，**显式传入**，kernel 不自动取 $1/\sqrt d$
- $\sigma_n$ = `sinks`（每 head 一个，只进分母，等价于 value=0 的虚拟 key）
- LSE $=m_i+\log Z_i$

### 6.2 压缩侧 $\kappa_i$

$$
L_{restored}=L_{cmp}\cdot r+\rho,\qquad
\kappa_i=\min\Big(\max\big(\lfloor (L_{restored}-L_q+i+1)/r\rfloor,0\big),L_{cmp}\Big)
$$

- $r$ = `cmp_ratio`，$\rho$ = `cmp_residual_kv`
- 向下取整是为了**因果安全**（只用完全覆盖的压缩 token）
- $\rho$ 补"除不尽的毛边"，不传会整体偏一个 token

### 6.3 三种场景

| 场景 | 输入特征 | $\mathcal{C}_i$ | `cmp_ratio`（A3） |
|---|---|---|---|
| SWA | 无 `cmp_kv` | $\varnothing$ | 1 |
| CSA | 有 `cmp_kv` + `cmp_sparse_indices` | TopK 选出的 token | 1 / 2 / 4 |
| HCA | 有 `cmp_kv`，无索引 | $[0,\kappa_i)$ | 128 |

**TopK 由外部 indexer 产生，本算子只消费/过滤索引。**

### 6.4 A3 硬约束（都写进了文档与 docstring）

| 参数 | 约束 |
|---|---|
| `num_heads_q` | 1/2/4/8/16/32/64/128 |
| `num_heads_kv` | 只能 1 |
| `head_dim` | 只能 512（nope 448 + rope 64） |
| `ori_win_left` / `ori_win_right` | **127 / 0** |
| `cmp_ratio` | SWA=1，CSA∈{1,2,4}，HCA=128 |
| `ori_sparse_indices` | 不支持（`ori_topk` 只能 0） |
| `topk_value_mode` | 只能 1 |
| `sinks` | **必传**（非空、一维、长度 N1、float32） |
| `cmp_residual_kv` | `cmp_mask_mode=3` 且 `cmp_ratio!=1` 时**必传** |
| `cmp_topk_length` | **保留字段，A2/A3 必须不传** |
| block_size | 16 的倍数，最大 1024 |

---

## 7. 已理解的关键机制

### 7.1 wheel 是什么

按约定组织的 zip：Python 源码 + 少量 `.so` + 数据文件 + `dist-info` 元数据。`pip install` 只是解压+登记，不编译不链接。

### 7.2 vendor 是什么

CANN 的"算子供应商包"：`op_proto/`（原型）+ `op_api/lib/libcust_opapi.so`（aclnn）+ `op_impl/ai_core/tbe/kernel/<soc>/*.o`（设备 kernel）+ `op_impl/cpu/`（AICPU）。

- 我们 wheel 里有两个：`sgl_kernel_npu`（本仓库算子）和 `vendor_sgl_ops_transformer_transformer`（上游算子）
- runtime 通过 `ASCEND_CUSTOM_OPP_PATH` 扫到它们
- **CANN 9.2.0 自带同名 `SparseFlashMla`**，导入顺序不对时会命中内置版本（见 8.2 坑 4）

### 7.3 host / device 两套"加载"

| | host 侧 `.so` | device 侧 `.o` |
|---|---|---|
| 跑在 | CPU | AICore / AICPU |
| 谁加载 | `dlopen`（`torch.ops.load_library` / `OpBuilder.load`） | CANN runtime + 驱动 |
| 何时 | import / 首次调用 | 算子执行时（首次上传后复用） |
| 找不到时 | `ImportError` / `OSError` | aclnn/tiling 报错、或命中错误实现 |

### 7.4 tiling key

把 10 个编译期模板参数（`FLASH_DECODE`、`LAYOUT_T`、`KV_LAYOUT_T`、`TEMPLATE_MODE`、`SPLIT_G`、`HEAD_RATIO_ONE`、`BATCH_CONSISTENCY`、`IS_VEC_S2PHYADDR`、`TOPK_VALUE_MODE`、`IS_DSPARK`）压成一个整数，host tiling 用 `GET_TPL_TILING_KEY(...)` 生成、`SetTilingKey()` 交给 runtime。

- **key 选"哪份编译好的 kernel"；tiling data 传"具体数字"**
- 合法组合由 `ASCENDC_TPL_SEL` 限定
- 这解释了一个 wheel 为什么能同时跑 SWA/CSA/HCA（不同 key → 不同模板实例）

### 7.5 metadata（调度表）

- `int32[2048]`，结构是 `faMetadata[80][9] + fdMetadata[80][8]`（1360 ≤ 2048）
- FA：cube 核负责的 `(BN2, M, S2)` 区间；FD：vector 核的合并任务
- 两段式 API：必须先算 metadata，且**调度参数必须与主算子完全一致**
- 沿 $S_2$ 切（split-K）才需要 FD 合并；沿 $M$ / $BN_2$ 切不需要

### 7.6 tile / 长方体 / 在线 softmax

- **tile** = $128\times128$（query × key，单 head）的索引方块；分数块 32 KiB（fp16）
- **长方体** = 分给一个核的 $(BN_2,\ M,\ S_2)$ 索引范围
- 窗口 128 让每个 query 块只碰 ~2 个 KV 块 → 约 97% 的 tile 是空的（"对角带"）
- **online softmax**：状态 $(m,\ \text{den},\ \text{acc})$ **按 (head, query 行)** 独立；沿 $S_2$ 分块累积；换 $m$ 时用 $e^{m_{old}-m_{new}}$ 同时缩放分子分母；最后 `acc/den`
- acc 大小 = $M_{tile}\times512\times4\text{B}$ = 256 KiB（$M_{tile}=128$）→ 超 L0C，故实际会折中 $M_{tile}$ 或分组 head

### 7.7 布局

| 布局 | 形状 | 含义 |
|---|---|---|
| `BSND` | `[B,S,N,D]` | 常规 padded |
| `TND` | `[T,N,D]` + `cu_seqlens` | 打包变长，零 padding |
| `PA_BBND` | `[页数, 页大小, N, D]` | PagedAttention KV 分页（配 `block_table`） |
| `PA_BNBD` | `[页数, N, 页大小, D]` | 分页的另一种顺序 |
| `PA_NZ` | `[页数,1,36,128,16]` | Ascend 专有分形布局 |

D 放最后是为了**向量连续**（512×2B = 1024 B 一次突发读完）；但计算单元还会再切成 16×16 fragment（甚至转置），且 `PA_NZ` 会故意偏离 D-last。

### 7.8 算术强度（讲 tile 时的量化依据）

| tile | FLOP | 搬运 | 强度 |
|---|---|---|---|
| 1×128（单 head） | 131 K | ~131 KB | ≈1 FLOP/B |
| 8×128 | 1.05 M | ~155 KB | ≈6.8 |
| 128×128 | 16.8 M | ~524 KB | ≈32 |

decode 时靠 64 个 head 共享 K tile 把强度拉回 ~26 FLOP/B，但并行度不足 → split-K。

---

## 8. 验证与踩坑记录

### 8.1 验证结果

- A3 上 6 个用例全部通过（`--skip-acc-check=false`，auto vs torch_native 对拍，bf16 `atol=rtol=5e-2`）
- 覆盖：SWA / CSA(2,4) / HCA(128) / 生产级 decode-CSA / BSND
- 未做：性能数据（设备被占用）、arch35 验证（本算子只注册 arch22）

### 8.2 四个坑（都已在代码/文档中固化）

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `Parameter sinks ... invalid. sinks must be provided` | `def.cpp` 标 OPTIONAL，但 tiling 无条件要求 | 测试所有用例传 `sinks`（float32 `[N1]`） |
| 2 | `Cmp_residual_kv is required when cmp_mask_mode=3 and cmp_ratio != 1` | CSA/HCA 必须传残差 | 传 `cmp_residual_kv`（int32 `[B]`，测试用 0），**metadata 也要传同一份** |
| 3 | `cmp_topk_length is reserved and does not support non-empty tensor on A2/A3` | A2/A3 上该字段保留 | 测试不传；文档标注 |
| 4 | `metadata dim 0 must be 1024`（我们传 2048） | 导入顺序晚于 `torch_npu` → CANN 只发现内置旧版算子 | 测试在模块级先 `import sgl_kernel_npu`；文档写清导入顺序要求 |

**坑 4 的机理**：`sgl_kernel_npu/__init__.py` 会把 vendor 路径写进 `ASCEND_CUSTOM_OPP_PATH`，但 CANN 在 `torch_npu` 初始化时就读这个变量。pytest 的 conftest 在 fixture 里才 import torch_npu，所以测试必须在 **collection 阶段**先 import 包。

---

## 9. 学习进度与待办

### 9.1 已经讲透（新会话不需要重讲，但可以复习）

1. wheel / vendor / host-device 三层加载模型
2. Python 注册表与 `__init__.py` / `__all__` / import 副作用
3. 调用链（Python 门面 → 上游绑定 → JIT → aclnn → kernel）
4. 算子数学：窗口、压缩阈值、sink、LSE
5. online softmax、$(m,\text{den},\text{acc})$、分块与 rescale
6. tile / 长方体 / split-K / 算术强度
7. tiling key 与模板特化
8. metadata 的作用与两段式调用
9. 布局 TND/BSND/PA_*

### 9.2 待讲（下一阶段）

1. `op_host/sparse_flash_mla_tiling.cpp`：参数 → block 数 / 模板模式 / tiling data
2. `op_kernel/sparse_flash_mla.cpp`：AIC/AIV 双核实例化的入口
3. `arch22/sparse_flash_mla_swa_kernel.h`：主循环骨架 `Process()`
4. `arch22/*_block_cube.h` 与 `*_block_vector.h`：cube/vector 分工与 GM workspace 交换
5. `Mmad` 层面的流水线（可选，按需）

### 9.3 教学例子（`csrc/examples/teaching/`，本分支新增）

从"读代码"切到"上机看实感"。例子在 `csrc/examples/teaching/`，注册成一个正式算子
（ID `teaching` / OpType `SglTeaching`），走 `opdef → tiling → kernel → aclnn → torch adapter`
这条链——也就是 sparse_flash_mla 进 wheel 的同一条链。

**为什么源码必须在 `csrc/` 下**：`csrc/cmake/cann_ops.cmake:198` 以 `csrc` 为基准算
kernel 目录的相对路径，CANN 的 `ascendc_copy_kernel_src.py` 要把 kernel 源码拷进
`autogen/`，它拒绝带 `..` 的路径（`Unsafe relative path`）。

| 级别 | 目录 | 概念 | 状态 |
|---|---|---|---|
| 01 | `csrc/examples/teaching/01_mmad_basic` | 一条 Mmad 的五个阶段（GM→L1→L0A/L0B→M→L0C→FIX→GM），16×16×16 | 代码完成，**待 NPU 验证** |
| 02 | `csrc/examples/teaching/02_mmad_k_loop` | K 分块累加：`cmatrixInitVal` / `unitFlag` | 待加 |
| 03 | `csrc/examples/teaching/03_mmad_shapes` | M/N 非 16 倍数、多 fractal、stride | 待加 |
| 04 | `csrc/examples/teaching/04_cube_vector` | cube→GM workspace→vector 的软流水（sparse_flash_mla 骨架） | 待加 |

调用链与正式算子一致（这也是 sparse_flash_mla 进 wheel 的方式）：

```
torch.ops.npu.<op> -> torch_adapter -> aclnn -> host tiling -> device kernel
```

**关键教训（踩过一次）**：`cmake/legacy_kernels.cmake` 只服务于老的 `build.sh`
（`BUILD_KERNELS_MODULE=ON`）。`setup.py` 的 wheel 由 `csrc/aclnn_torch_adapter/CMakeLists.txt`
构建，它**只编译被 `--ops` 选中算子的 `TORCH_ADAPTER`**，根本不 include legacy 列表。
所以往 legacy 列表加东西 → 编不进 wheel → `torch.ops.npu.xxx` 报 no attribute。

**所有教学例子共用一个算子 ID**：CANN / torch 两级构建缓存都以 `--ops` 选择串为 key
（`build_tools/cann.py:154`、`torch_wheel.py:57`），新增 ID 就要 `--make-clean` 全量重来。
换例子只改声明文件里的 4 个源路径，选择串恒为 `apply_token_bitmask,teaching`。

构建/运行（首次需清缓存）：

```bash
python3 setup.py --make-clean
python3 setup.py --arch-version 2201 --ops apply_token_bitmask,teaching -j 16
pip install --no-index --no-deps --force-reinstall output/sgl_kernel_npu-*.whl
python3 csrc/examples/teaching/01_mmad_basic/run.py
```

### 9.4 待办事项

- [ ] NPU 空闲时补性能数据（`--enable-perf=true --csv-path=...`，跑 2–3 次看波动）
- [ ] 写 PR 描述（目标分支 `main`；附 4 个 A3 约束、6 用例结果）
- [ ] 讨论 commit pin：`7df757ad`（会议指定）vs `37c98ef`（对 arch35 `flash_mla` 零影响）
- [ ] 讨论「CANN 内置同名算子」与「打包私有 vendor」的共存策略
- [ ] （可选）给测试加一条断言：`ASCEND_CUSTOM_OPP_PATH` 第一项必须是 wheel 内 vendor

---

## 10. 常用命令速查

```bash
# 包在哪、manifest 是什么
python3 -c "import pathlib,sgl_kernel_npu;print(pathlib.Path(sgl_kernel_npu.__file__).parent)"
python3 -c "import json,pathlib,sgl_kernel_npu;r=pathlib.Path(sgl_kernel_npu.__file__).parent;print(json.dumps(json.loads((r/'_build_manifest.json').read_text())['third_party'],indent=2))"

# vendor / OPP_PATH
ls $(python3 -c "import pathlib,sgl_kernel_npu;print(pathlib.Path(sgl_kernel_npu.__file__).parent)")/_cann/vendors
python3 -c "import os,sgl_kernel_npu;print(os.environ['ASCEND_CUSTOM_OPP_PATH'])"

# 注册表
python3 -c "
import sgl_kernel_npu.operators.attention
from sgl_kernel_npu.operators.utils.registry import list_impls_by_name
print(list_impls_by_name('attention.sparse_flash_mla'))
"

# 上游函数落在哪
python3 -c "
import importlib,inspect
m=importlib.import_module('sgl_kernel_npu.operators.attention.sparse_flash_mla')
op,meta=m._load_cann_fns()
print(op.__module__); print(inspect.getsourcefile(op))
"

# JIT 缓存
ls ~/.cache/torch_extensions/sparse_flash_mla/

# 构建缓存
jq -r '.fingerprint, .inputs.config.commit' build/third_party/cache/*.json
```

---

## 11. 与新会话的协作约定

### 11.1 用户背景

- 做 **SGLang 框架开发/优化**，对算子实现是**新手**（但理解能力很好，会主动推演）
- 会追问"为什么这样设计""这个数字怎么来的""我这个理解对不对"
- 喜欢**具体数字**和**代码位置**，不喜欢空泛描述

### 11.2 讲解风格（上一会话验证有效）

1. **先给直觉/类比，再给公式，最后落到代码**
2. 每讲一个量，都用**具体数字**验证（如 $L_{kv}=128, L_q=16, i=0 \rightarrow \theta=113$）
3. 明确"**哪个是已知输入、哪个是推导**"
4. 主动指出用户理解中的**偏差**，并给出修正版表述
5. 引用代码时给**文件:行号**
6. 不要一次抛太多概念；一个概念闭环后再下一个

### 11.3 不要做的事

- 不要修改 `feat/sparse-flash-mla` 分支（那是待提 PR 的代码）
- 本学习分支只加**文档**，不改算子代码
- 不要假设 Mac 上有 NPU；需要跑设备命令时**请用户执行并回贴输出**
- 不要把材料放 `/tmp`（会被清理）
- 不要凭记忆断言上游实现细节；以 `/Users/an/Developer/ops-transformer` 源码为准（注意它是 `master`，pin 是 `7df757ad`）

---

## 12. 术语表

| 术语 | 含义 |
|---|---|
| ops-transformer | CANN 的算子仓库（gitcode），算子本体所在地 |
| vendor | CANN 的算子供应商包目录 |
| tiling | host 侧把 shape/参数翻译成 block 数、workspace、kernel 选择 |
| tiling key | 模板参数编码，选 kernel 实例 |
| metadata | `int32[2048]` 调度表（FA/FD 两段） |
| tile | 128×128 的计算方块 |
| cuboid | 分给一个核的 $(BN_2, M, S_2)$ 范围 |
| split-K | 沿 $S_2$ 切分并行，需要合并 |
| FA / FD | cube 核任务 / vector 核合并任务 |
| online softmax | 分块累加 $(m,\text{den},\text{acc})$ 的精确 softmax |
| PA | PagedAttention（KV 分页） |
| CSA / HCA / SWA | 压缩稀疏 / 重度压缩 / 滑动窗口三种场景 |
| AICPU | 设备上的 CPU 核（跑 metadata 这类轻量逻辑） |
| NZ | Ascend 分形布局 |
