# 02 — K 方向分块累加

契约和 01 相同：`C[M,N] = A[M,K] · B[N,K]ᵀ`。变的是 **K**：kernel 沿 K 每 16 个元素做一条
Mmad，全部累加进同一个 L0C。

**02 是 01 的超集**：`K = 16` 时 `kLoop = 1`，行为与 01 完全一致。

## 与 01 的差异只有四处

| # | 01 | 02 | 为什么要改 |
|---|---|---|---|
| 1 | tiling data 4 B（只有 `blockDim`） | 12 B（+`kLoop` / `kTotal`），**kernel 真的读** | 同一个 kernel 要服务不同 K，循环次数必须由 host 传 |
| 2 | `srcDValue = 16`（= `dValue`，紧凑） | `srcDValue = K`（**> dValue**，strided） | 每次只取 16 列，但下一行隔着 K 个元素 |
| 3 | 一条 Mmad，`cmatrixInitVal=true`、`unitFlag=0b11` | 循环里 `k==0` 时 true、其余 false；最后一条 0b11、其余 0b10 | 累加语义 + "什么时候能搬走" |
| 4 | 无 | **新增两条反向同步** `MTE1→MTE2`、`M→MTE1` | 循环会反复覆盖 L1 和 L0A/L0B |

## 文件地图

| 文件 | 与 01 的关系 |
|---|---|
| `op_kernel/mmad_kloop_kernel.cpp` | 新增循环 + 反向同步（核心） |
| `op_kernel/mmad_kloop_tiling.h` | 3 个字段（01 是 1 个） |
| `op_host/mmad_kloop_op.cpp` | tiling 从 15 行变 30 行（算 `kLoop`） |
| `op_host/torch_adapter.cpp` | **与 01 完全一致**——契约没变，接口层不用动 |
| `run.py` | K ∈ {16, 64, 128, 512} 全部对拍 |

> `01_mmad_basic/` 的源码仍保留在仓库里作为阅读材料，但**当前编译的是 02**
> （`csrc/examples/teaching/CMakeLists.txt` 指向 02）。

## 四条同步边，四个事件槽位

```
MTE2 ──(数据好了)────> MTE1   EVT_DATA_READY = 0   正向
MTE1 ──(L1 空了)─────> MTE2   EVT_BUF_FREE   = 1   反向 ← 02 新增
MTE1 ──(操作数好了)──> M     EVT_OP_READY   = 2   正向
M    ──(L0 空了)─────> MTE1   EVT_L0_FREE    = 3   反向 ← 02 新增
```

01 里只有"正向"两条就够了，因为每块 buffer 只用一次。02 进入循环后，**同一块
buffer 会被反复覆盖**，所以必须补两条反向同步：

| 漏掉哪条 | 后果 |
|---|---|
| `MTE1→MTE2`（L1 空） | 下一轮 `DataCopy` 覆盖还在被 `LoadData` 读的 L1 |
| `M→MTE1`（L0 空） | 下一轮 `LoadData` 覆盖还在被 `Mmad` 读的 L0A/L0B |

两者都是"偶发错乱"，跑一次可能还对。

陷阱：第一轮没有"上一轮"可等，所以循环前要**先 `SetFlag` 放行一次**，否则第一个
`WaitFlag` 会永远等下去（`mmad_kloop_kernel.cpp` 的 `Process()` 开头那两行）。

## Mmad 累加的全部秘密

```cpp
mm.cmatrixInitVal = (k == 0);                           // 第 1 条清零，其余累加
mm.unitFlag = (k + 1 == kLoop_) ? 0b11 : 0b10;          // 只有最后一条允许搬出
Mmad(c, a2, b2, mm);
PipeBarrier<PIPE_M>();
```

`kLoop = 32`（K=512）时，就是 32 条 Mmad 累加进同一个 L0C，最后 Fixpipe 搬一次。
**这就是 attention 里 `S = Q·Kᵀ` 沿 head_dim=512 累加的真实形态。**

## 预期输出

```
A: [16, K] fp16   B: [16, K] fp16   C: [16, 16] fp32   (B 是 [N,K])
同一份 kernel，不同 K，只靠 tiling data 里的 kLoop / kTotal 驱动

K=  16  kLoop= 1  Mmad条数= 1  max|c-ref|=x.xxx e-06  PASS  (= 例子 01)
K=  64  kLoop= 4  Mmad条数= 4  max|c-ref|=x.xxx e-06  PASS
K= 128  kLoop= 8  Mmad条数= 8  max|c-ref|=x.xxx e-06  PASS
K= 512  kLoop=32  Mmad条数=32  max|c-ref|=x.xxx e-06  PASS
```

## 编译 & 运行

```bash
cd /home/a00821909/sgl-kernel-npu
git pull --ff-only origin learn/sparse-flash-mla
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3 setup.py --arch-version 2201 --ops apply_token_bitmask,teaching -j 16
pip install --no-index --no-deps --force-reinstall output/sgl_kernel_npu-*.whl
python3 csrc/examples/teaching/02_mmad_k_loop/run.py
```

**不需要 `--make-clean`**：算子 ID 和选择串都没变，只是源文件换了，增量重编即可。

## 排错表

| 症状 | 最可能的原因 |
|---|---|
| 只有第 1 个 K 块正确（其余像没算） | `cmatrixInitVal` 一直 true |
| 结果大约是正确答案的 1/kLoop | `unitFlag` 一直是 0b11，Fixpipe 提前搬了 |
| 偶发错乱 / 结果随机 | 少了反向同步（`EVT_BUF_FREE` 或 `EVT_L0_FREE`） |
| 第一轮就跑飞 / 卡死 | 循环前少了那个"开局放行"的 `SetFlag` |
| 结果整体偏小或错位 | `srcDValue` 没设成 K（strided 读错行） |
| `K must be multiple of 16` | 输入 K 没对齐（tiling 里会挡） |
