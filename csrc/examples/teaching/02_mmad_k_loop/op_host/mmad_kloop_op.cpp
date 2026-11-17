// 教学例子 02 的 host 侧：一个文件同时扮演三个角色
//   OPDEF_SOURCES / INFER_SOURCES / TILING_SOURCES
//
// 与例子 01 相比，opdef 和 infershape 完全没变（契约不变：C = A·Bᵀ），
// 只有 tiling 变了：现在它要按 K 的大小算出"循环几次"，并写进 tiling data。

#include "../op_kernel/mmad_kloop_tiling.h" // host 与 kernel 共享的 tiling 结构
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"            // 构建期生成的 SGL_CANN_CONFIGS
#include "tiling/platform/platform_ascendc.h" // gert::TilingContext

#include <cstdint>

namespace optiling
{

constexpr int64_t TILE_K = 16; // 与 kernel 侧保持一致：一条 Mmad 吃 16 个 K 元素

// ---- 角色 3：tiling ---------------------------------------------------------
// 和例子 01 的 15 行相比，多出来的就是"按 K 算循环次数"。
// 真实算子（sparse_flash_mla）的 tiling 也是在做同一件事，只是要算的量多得多：
// 分块大小、perfMode、workspace、tiling key …… 全部来自 shape/属性。
static ge::graphStatus TilingMmadKLoop(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (raw == nullptr || workspace == nullptr || raw->GetCapacity() < sizeof(SglTeachingTilingData)) {
        return ge::GRAPH_FAILED;
    }

    const auto *aShape = context->GetInputShape(0);
    if (aShape == nullptr || aShape->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const int64_t kTotal = aShape->GetDim(1); // A 是 [M, K]，第 1 维就是 K
    if (kTotal <= 0 || kTotal % TILE_K != 0) {
        return ge::GRAPH_FAILED; // K 必须对齐到 16，否则一个 fractal 都凑不齐
    }

    auto *tiling = context->GetTilingData<SglTeachingTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->blockDim = 1;                              // 仍然只用 1 个 AI Core
    tiling->kTotal = static_cast<uint32_t>(kTotal);     // GM 行间隔
    tiling->kLoop = static_cast<uint32_t>(kTotal / TILE_K); // 循环几次
    raw->SetDataSize(sizeof(SglTeachingTilingData));

    context->SetBlockDim(tiling->blockDim);
    context->SetTilingKey(0);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops
{

// ---- 角色 2：shape/dtype 推导 ------------------------------------------------
// 注意：InferShape 里 GetInputShape() 直接返回 gert::Shape*，它自己就是 shape，
// 没有 GetStorageShape()（那是 CompileTimeTensorDesc 那边的方法）。
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *a = context->GetInputShape(0);
    const auto *b = context->GetInputShape(1);
    auto *c = context->GetOutputShape(0);
    if (a == nullptr || b == nullptr || c == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (a->GetDimNum() != 2 || b->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape outShape;
    outShape.AppendDim(a->GetDim(0)); // M = A 的行
    outShape.AppendDim(b->GetDim(0)); // N = B 的行（B 是 [N,K]）
    *c = outShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // 输入 fp16，输出 fp32：cube 内部按 fp32 累加
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

// ---- 角色 1：算子原型 -------------------------------------------------------
// 输入 a[M,K]（fp16）、b[N,K]（fp16），输出 c[M,N]（fp32）：C = A·Bᵀ。
// 和例子 01 完全一致 —— K 的大小是运行时决定的，不需要改原型。
class SglTeaching : public OpDef
{
public:
    explicit SglTeaching(const char *name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(InferShape).SetInferDataType(InferDataType);
        this->AICore().SetTiling(optiling::TilingMmadKLoop);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglTeaching);

} // namespace ops
