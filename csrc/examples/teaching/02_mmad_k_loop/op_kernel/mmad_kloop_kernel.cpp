// 教学例子 02：K 方向分块累加
//
//   契约不变：C[M,N] = A[M,K] · B[N,K]ᵀ，A 是 [M,K]，B 是 [N,K]
//   唯一的变化：K 不再固定 16，而是"沿 K 切块、逐块累加"，这正是 attention 里
//   S = Q·Kᵀ 沿 head_dim = 512 累加的真实形态。
//
// 与例子 01 的差别只有四处：
//
//   1. tiling data 从 4 B 变成 12 B，而且 kernel 真的读它（kLoop / kTotal）
//   2. GM -> L1 变成 strided 读：每次只取 16 列，行间跨 kTotal 个元素
//   3. Mmad 进循环：cmatrixInitVal 第一条 true、其余 false；
//      unitFlag 最后一条 0b11、其余 0b10
//   4. 新增两条反向同步：MTE1->MTE2（L1 可覆盖）、M->MTE1（L0A/L0B 可覆盖）
//
// 当 K = 16 时 kLoop = 1，行为与例子 01 完全相同 —— 这就是"超集"的意思。

#include "kernel_operator.h"

#include "mmad_kloop_tiling.h"

namespace
{
using namespace AscendC;

constexpr uint32_t TILE_M = 16;
constexpr uint32_t TILE_N = 16;
constexpr uint32_t TILE_K = 16; // 每一次 Mmad 吃掉 16 个 K 元素（一个 fractal）

// 四个事件槽位，各管一条边，互不干扰（硬件一共 8 个）
constexpr int32_t EVT_DATA_READY = 0; // MTE2 -> MTE1：L1 里的数据好了（正向）
constexpr int32_t EVT_BUF_FREE = 1;   // MTE1 -> MTE2：L1 的 buffer 用完了（反向）
constexpr int32_t EVT_OP_READY = 2;   // MTE1 -> M   ：L0A/L0B 里的操作数好了（正向）
constexpr int32_t EVT_L0_FREE = 3;    // M    -> MTE1：cube 用完 L0A/L0B 了（反向）

class MmadKLoopKernel
{
public:
    static constexpr IsResetLoad3dConfig LOAD3D_CFG = {true, true}; // isSetFMatrix / isSetPadding

    __aicore__ inline MmadKLoopKernel() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t kTotal, uint32_t kLoop)
    {
        kTotal_ = kTotal; // GM 里每行有多少个 K 元素 = 行间隔
        kLoop_ = kLoop;   // 要循环几次 Mmad

        // 注意这里的 size 用的是 kTotal：整个 [M,K] 张量都在 GM 上，我们每次只取其中 16 列
        aGm_.SetGlobalBuffer((__gm__ half *)a, TILE_M * kTotal_);
        bGm_.SetGlobalBuffer((__gm__ half *)b, TILE_N * kTotal_);
        cGm_.SetGlobalBuffer((__gm__ float *)c, TILE_M * TILE_N);

        // 片上 buffer 仍然只要"一块"的大小：每次只搬一个 16 列的 K 块
        pipe_.InitBuffer(a1Buf_, TILE_M * TILE_K * sizeof(half));
        pipe_.InitBuffer(b1Buf_, TILE_N * TILE_K * sizeof(half));
        pipe_.InitBuffer(a2Buf_, TILE_M * TILE_K * sizeof(half));
        pipe_.InitBuffer(b2Buf_, TILE_N * TILE_K * sizeof(half));
        pipe_.InitBuffer(cBuf_, TILE_M * TILE_N * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // 两条反向同步的"开局放行"：第一轮没有上一轮可等，先自己置一次位，
        // 否则第一次 WaitFlag 会永远等下去。
        SetFlag<HardEvent::MTE1_MTE2>(EVT_BUF_FREE); // L1 空闲
        SetFlag<HardEvent::M_MTE1>(EVT_L0_FREE);     // L0A/L0B 空闲

        for (uint32_t k = 0; k < kLoop_; ++k) {
            // (a) 等 L1 空闲 —— 上一轮的 MTE1 读完没有
            WaitFlag<HardEvent::MTE1_MTE2>(EVT_BUF_FREE);

            // (b) GM -> L1：只取第 k 个 16 列的块（strided 读）
            CopyIn(k);

            // (c) 正向同步：L1 数据好了，MTE1 才能读
            SetFlag<HardEvent::MTE2_MTE1>(EVT_DATA_READY);
            WaitFlag<HardEvent::MTE2_MTE1>(EVT_DATA_READY);

            // (d) 等 cube 用完 L0A/L0B（上一轮的 Mmad 可能还在读）
            //     漏了这条 = 用新操作数覆盖正在被读的 L0，结果偶发错乱
            WaitFlag<HardEvent::M_MTE1>(EVT_L0_FREE);

            // (e) L1 -> L0A/L0B：仍然是"一个 16x16 fractal"
            LoadL1ToL0();

            // (f) 告诉 MTE2：L1 我读完了，下一轮可以覆盖
            SetFlag<HardEvent::MTE1_MTE2>(EVT_BUF_FREE);

            // (g) 正向同步：L0A/L0B 好了，cube 才能算
            SetFlag<HardEvent::MTE1_M>(EVT_OP_READY);
            WaitFlag<HardEvent::MTE1_M>(EVT_OP_READY);

            // (h) 累加这一块
            MmadTile(k);

            // (i) 告诉 MTE1：cube 用完了，下一轮可以覆盖 L0A/L0B
            SetFlag<HardEvent::M_MTE1>(EVT_L0_FREE);
        }

        // 循环结束，把 L0C 里的最终结果搬回 GM
        CopyOutL0cToGm();
    }

private:
    // ---- GM -> L1：只取第 k 个 K 块（本例最值得看的一步）-------------------
    __aicore__ inline void CopyIn(uint32_t k)
    {
        LocalTensor<half> a1 = a1Buf_.Get<half>();
        Nd2NzParams pa;
        pa.ndNum = 1;
        pa.nValue = TILE_M;                  // 16 行
        pa.dValue = TILE_K;                  // 每行只搬 16 个元素（连续）
        pa.srcDValue = (uint16_t)kTotal_;    // ← 但下一行要跨 kTotal 个元素！
        pa.dstNzC0Stride = TILE_M;           // L1 里一个 C0 块跨 16 行
        pa.dstNzNStride = 1;
        pa.srcNdMatrixStride = 0;
        pa.dstNzMatrixStride = 0;
        // 基址偏移 = k * 16：第 k 个块的起始列
        DataCopy(a1, aGm_[k * TILE_K], pa);

        LocalTensor<half> b1 = b1Buf_.Get<half>();
        Nd2NzParams pb;
        pb.ndNum = 1;
        pb.nValue = TILE_N; // 16 行 = N
        pb.dValue = TILE_K; // 每行 16 个 K
        pb.srcDValue = (uint16_t)kTotal_;
        pb.dstNzC0Stride = TILE_N;
        pb.dstNzNStride = 1;
        pb.srcNdMatrixStride = 0;
        pb.dstNzMatrixStride = 0;
        DataCopy(b1, bGm_[k * TILE_K], pb);
    }

    // ---- L1 -> L0A/L0B：与例子 01 完全一致（因为 L1 里永远只有一个 16x16 块）--
    __aicore__ inline void LoadL1ToL0()
    {
        LocalTensor<half> a1 = a1Buf_.Get<half>();
        LocalTensor<half> b1 = b1Buf_.Get<half>();
        LocalTensor<half> a2 = a2Buf_.Get<half>();
        LocalTensor<half> b2 = b2Buf_.Get<half>();

        LoadData3DParamsV2<half> ldA;
        ldA.l1H = TILE_M / 16;
        ldA.l1W = 16;
        ldA.padList[0] = 0;
        ldA.padList[1] = 0;
        ldA.padList[2] = 0;
        ldA.padList[3] = 0;
        ldA.mExtension = TILE_M;
        ldA.kExtension = TILE_K;
        ldA.mStartPt = 0;
        ldA.kStartPt = 0;
        ldA.strideW = 1;
        ldA.strideH = 1;
        ldA.filterW = 1;
        ldA.filterSizeW = 0;
        ldA.filterH = 1;
        ldA.filterSizeH = 0;
        ldA.dilationFilterW = 1;
        ldA.dilationFilterH = 1;
        ldA.enTranspose = 0;
        ldA.fMatrixCtrl = 0;
        ldA.channelSize = TILE_K;
        LoadData<half, LOAD3D_CFG>(a2, a1, ldA);

        LoadData2DParams ldB;
        ldB.startIndex = 0;
        ldB.repeatTimes = (TILE_N / 16) * (TILE_K / 16); // 1 个 fractal
        ldB.srcStride = 1;
        ldB.dstGap = 0;
        ldB.ifTranspose = false;
        LoadData(b2, b1, ldB);
    }

    // ---- Mmad：这里就是"累加"的全部秘密 -----------------------------------
    __aicore__ inline void MmadTile(uint32_t k)
    {
        LocalTensor<half> a2 = a2Buf_.Get<half>();
        LocalTensor<half> b2 = b2Buf_.Get<half>();
        LocalTensor<float> c = cBuf_.Get<float>();

        MmadParams mm;
        mm.m = TILE_M;
        mm.n = TILE_N;
        mm.k = TILE_K;
        // 第 0 块：C = A·Bᵀ（覆盖 L0C 里的脏数据）
        // 之后  ：C = C + A·Bᵀ（在上一次的结果上累加）
        mm.cmatrixInitVal = (k == 0);
        mm.cmatrixSource = false;
        // 最后一块：告诉硬件"我完事了，Fixpipe 可以搬"；中间块不许搬
        mm.unitFlag = (k + 1 == kLoop_) ? 0b11 : 0b10;
        Mmad(c, a2, b2, mm);
        PipeBarrier<PIPE_M>(); // 小 tile（1 个 fractal）必须打屏障，见上游同款做法
    }

    // ---- L0C -> GM：循环全部结束后只做一次 ---------------------------------
    __aicore__ inline void CopyOutL0cToGm()
    {
        LocalTensor<float> c = cBuf_.Get<float>();
        FixpipeParamsV220 fp;
        fp.nSize = TILE_N;
        fp.mSize = TILE_M;
        fp.srcStride = TILE_M;
        fp.dstStride = TILE_N;
        fp.unitFlag = 0b11; // 与最后一条 Mmad 的 unitFlag 配对
        fp.ndNum = 1;
        Fixpipe(cGm_, c, fp);
    }

    TPipe pipe_;
    TBuf<TPosition::A1> a1Buf_;
    TBuf<TPosition::B1> b1Buf_;
    TBuf<TPosition::A2> a2Buf_;
    TBuf<TPosition::B2> b2Buf_;
    TBuf<TPosition::CO1> cBuf_;
    GlobalTensor<half> aGm_;
    GlobalTensor<half> bGm_;
    GlobalTensor<float> cGm_;
    uint32_t kTotal_ = 0;
    uint32_t kLoop_ = 0;
};

} // namespace

// 设备侧入口：名字仍是 OpType 的 snake_case（SglTeaching -> sgl_teaching）。
// 这次 tiling data 真的被用上了：kLoop / kTotal 决定循环几次、行间隔多少。
extern "C" __global__ __aicore__ void sgl_teaching(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace,
                                                  GM_ADDR tiling)
{
    (void)workspace;

    REGISTER_TILING_DEFAULT(optiling::SglTeachingTilingData);
    GET_TILING_DATA(data, tiling);

    MmadKLoopKernel op;
    op.Init(a, b, c, data.kTotal, data.kLoop);
    op.Process();
}
