// 教学例子 01：一条 Mmad 到底干了什么
//
//   C[16,16](fp32) = A[16,16](fp16) @ B[16,16](fp16)
//
// 整条数据通路就是 Mmad 的五个阶段，缺一不可：
//
//   GM --MTE2(DataCopy+Nd2Nz)--> L1 --MTE1(LoadData)--> L0A / L0B
//      --M(Mmad)--> L0C --FIX(Fixpipe)--> GM
//
// 选择 16x16x16 是因为它正好是 cube 的一个最小"分形"(fractal)：
//   - fp16 的 C0 = 16，所以一个 fractal = 16x16 个 fp16 = 512 B
//   - L0A/L0B/L0C 都按 512 B 粒度组织，一个 fractal 刚好不浪费一行
// 也就是说：这个例子里，每个阶段都只搬 / 只算"一块"，没有任何循环。

#include "kernel_operator.h"

namespace
{
using namespace AscendC;

constexpr uint32_t TILE_M = 16;
constexpr uint32_t TILE_N = 16;
constexpr uint32_t TILE_K = 16;

// isSetFMatrix / isSetPadding 都交给 LoadData 自己做（省掉手写 SetFmatrix）
constexpr IsResetLoad3dConfig LOAD3D_CFG = {true, true};

class MmadBasicKernel
{
public:
    __aicore__ inline MmadBasicKernel() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c)
    {
        // blockDim=1，所以这里不需要按 GetBlockIdx() 做偏移；
        // 真实 kernel（sparse_flash_mla）里每个核会 SetGlobalBuffer(基址 + 自己的偏移)。
        aGm_.SetGlobalBuffer((__gm__ half *)a, TILE_M * TILE_K);
        bGm_.SetGlobalBuffer((__gm__ half *)b, TILE_K * TILE_N);
        cGm_.SetGlobalBuffer((__gm__ float *)c, TILE_M * TILE_N);

        // 每块 buffer 的字节数 = 一个 fractal
        pipe_.InitBuffer(a1Buf_, TILE_M * TILE_K * sizeof(half));   // L1  里的 A
        pipe_.InitBuffer(b1Buf_, TILE_K * TILE_N * sizeof(half));   // L1  里的 B
        pipe_.InitBuffer(a2Buf_, TILE_M * TILE_K * sizeof(half));   // L0A 里的 A
        pipe_.InitBuffer(b2Buf_, TILE_K * TILE_N * sizeof(half));   // L0B 里的 B
        pipe_.InitBuffer(cBuf_, TILE_M * TILE_N * sizeof(float));   // L0C 里的 C
    }

    __aicore__ inline void Process()
    {
        CopyInGmToL1();
        // MTE2 往 L1 写完，MTE1 才能从 L1 读
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        LoadL1ToL0();
        // MTE1 往 L0A/L0B 写完，cube(M) 才能读
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);

        MmadOneTile();
        CopyOutL0cToGm();
    }

private:
    // ---- 阶段 1：GM -> L1（MTE2）------------------------------------------
    // GM 里是普通行主序 ND；L1 里必须是 cube 认识的 nZ 分形布局，
    // 所以不能直接 DataCopy，要用 Nd2NzParams 让硬件顺手做这个重排。
    __aicore__ inline void CopyInGmToL1()
    {
        LocalTensor<half> a1 = a1Buf_.Get<half>();
        Nd2NzParams pa;
        pa.ndNum = 1;              // 只有 1 个矩阵
        pa.nValue = TILE_M;        // 16 行
        pa.dValue = TILE_K;        // 每行 16 列
        pa.srcDValue = TILE_K;     // GM 里行间隔 = 16 个元素（紧凑 ND）
        pa.dstNzC0Stride = TILE_M; // L1 里一个 C0 块跨 16 行
        pa.dstNzNStride = 1;
        pa.srcNdMatrixStride = 0;
        pa.dstNzMatrixStride = 0;
        DataCopy(a1, aGm_, pa);

        LocalTensor<half> b1 = b1Buf_.Get<half>();
        Nd2NzParams pb;
        pb.ndNum = 1;
        pb.nValue = TILE_K;        // 16 行（B 的行是 K）
        pb.dValue = TILE_N;        // 每行 16 列（B 的列是 N）
        pb.srcDValue = TILE_N;
        pb.dstNzC0Stride = TILE_K;
        pb.dstNzNStride = 1;
        pb.srcNdMatrixStride = 0;
        pb.dstNzMatrixStride = 0;
        DataCopy(b1, bGm_, pb);
    }

    // ---- 阶段 2：L1 -> L0A / L0B（MTE1）-----------------------------------
    // L1 是"大仓库"，L0A/L0B 是紧贴 cube 的寄存器阵列。
    // 矩阵乘的每个操作数都要从 L1 再搬一次到 L0A/L0B，这一步才是 MTE1 的活。
    __aicore__ inline void LoadL1ToL0()
    {
        LocalTensor<half> a1 = a1Buf_.Get<half>();
        LocalTensor<half> b1 = b1Buf_.Get<half>();
        LocalTensor<half> a2 = a2Buf_.Get<half>();
        LocalTensor<half> b2 = b2Buf_.Get<half>();

        // A：从 L1 的 nZ 布局切成 L0A 的 zZ 布局（16x16 一个块）
        LoadData3DParamsV2<half> ldA;
        ldA.l1H = TILE_M / 16;   // L1 里 M 方向有几个 16 行块 → 1
        ldA.l1W = 16;            // 每个块宽 M0 = 16
        ldA.padList[0] = 0;
        ldA.padList[1] = 0;
        ldA.padList[2] = 0;
        ldA.padList[3] = 0;
        ldA.mExtension = TILE_M; // 真正要取 M = 16
        ldA.kExtension = TILE_K; // 真正要取 K = 16
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
        ldA.enTranspose = 0;     // 不转置：A 是 (M,K)
        ldA.fMatrixCtrl = 0;
        ldA.channelSize = TILE_K;
        LoadData<half, LOAD3D_CFG>(a2, a1, ldA);

        // B：从 L1 的 nZ 直接按 16x16 fractal 搬进 L0B
        LoadData2DParams ldB;
        ldB.startIndex = 0;
        ldB.repeatTimes = (TILE_N / 16) * (TILE_K / 16); // 1 个 fractal
        ldB.srcStride = 1;
        ldB.dstGap = 0;
        ldB.ifTranspose = false;
        LoadData(b2, b1, ldB);
    }

    // ---- 阶段 3：Mmad（cube）---------------------------------------------
    // 关键三个数字：m/n/k 只描述"这一次乘加", 与整体 shape 无关。
    // 一条 16x16x16 的 Mmad = 16*16*16 = 4096 次乘加。
    __aicore__ inline void MmadOneTile()
    {
        LocalTensor<half> a2 = a2Buf_.Get<half>();
        LocalTensor<half> b2 = b2Buf_.Get<half>();
        LocalTensor<float> c = cBuf_.Get<float>();

        MmadParams mm;
        mm.m = TILE_M;
        mm.n = TILE_N;
        mm.k = TILE_K;
        // true  = C = A*B      （初始化 C，覆盖 L0C 里的旧值）
        // false = C = C + A*B  （累加，K 方向分块的第二次之后用这个）
        mm.cmatrixInitVal = true;
        mm.cmatrixSource = false;
        // 0b11 = 这条 Mmad 是"最后一条", 硬件置位后 Fixpipe 才能搬
        mm.unitFlag = 0b11;
        Mmad(c, a2, b2, mm);
        PipeBarrier<PIPE_M>();
    }

    // ---- 阶段 4：L0C -> GM（Fixpipe）--------------------------------------
    // L0C 里的结果是 fp32 的分形布局，必须靠 Fixpipe 才能变回 ND 写回 GM。
    __aicore__ inline void CopyOutL0cToGm()
    {
        LocalTensor<float> c = cBuf_.Get<float>();
        FixpipeParamsV220 fp;
        fp.nSize = TILE_N;       // 输出 N = 16
        fp.mSize = TILE_M;       // 输出 M = 16
        fp.srcStride = TILE_M;   // 本例 M=N=16，两种口径都等于 16
        fp.dstStride = TILE_N;   // GM 里相邻行间隔 = 16 个 float
        fp.unitFlag = 0b11;      // 与上面 Mmad 的 unitFlag 配对
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
};

} // namespace

// 设备侧入口：名字叫 mmad_basic，CMake 会据此生成 aclrtlaunch_mmad_basic.h
extern "C" __global__ __aicore__ void mmad_basic(GM_ADDR a, GM_ADDR b, GM_ADDR c)
{
    MmadBasicKernel op;
    op.Init(a, b, c);
    op.Process();
}
