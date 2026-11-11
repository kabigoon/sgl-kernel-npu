// 教学例子 01 的 host 侧：一个文件同时扮演三个角色
//   OPDEF_SOURCES / INFER_SOURCES / TILING_SOURCES
//
// sparse_flash_mla 里这三件事分在三个文件（def.cpp / infershape.cpp / tiling.cpp），
// 这里合在一起，是因为例子小到能一屏看完。

#include "../op_kernel/mmad_basic_tiling.h"          // host 与 kernel 共享的 tiling 结构
#include "register/op_def_registry.h"
#include "sgl_operator_config.h"                    // 构建期生成的 SGL_CANN_CONFIGS

#include <cstdint>

namespace optiling
{

// ---- 角色 3：tiling ---------------------------------------------------------
// 这就是"参数 → block 数 / tiling data"那条链的最小形态。
// sparse_flash_mla 的同名函数有 130 行：它要根据 shape 算 workspace、算
// perfMode、算 mBaseSize、最后拼 tiling key；这里只需要说"开 1 个核"。
static ge::graphStatus TilingMmadBasic(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto *raw = context->GetRawTilingData();
    auto *workspace = context->GetWorkspaceSizes(1);
    if (raw == nullptr || workspace == nullptr || raw->GetCapacity() < sizeof(SglTeachingTilingData)) {
        return ge::GRAPH_FAILED;
    }

    auto *tiling = context->GetTilingData<SglTeachingTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->blockDim = 1;  // 例子 01 只用 1 个 AI Core
    raw->SetDataSize(sizeof(SglTeachingTilingData));

    context->SetBlockDim(tiling->blockDim);
    context->SetTilingKey(0);  // 只有一个 kernel 实例，key 恒定
    workspace[0] = 0;          // kernel 不需要 workspace
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ops
{

// ---- 角色 2：shape/dtype 推导 ------------------------------------------------
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *a = context->GetInputShape(0);
    const auto *b = context->GetInputShape(1);
    auto *c = context->GetOutputShape(0);
    if (a == nullptr || b == nullptr || c == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &sa = a->GetStorageShape();
    const auto &sb = b->GetStorageShape();
    if (sa.GetDimNum() != 2 || sb.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    c->SetDimNum(2);
    c->SetDim(0, sa.GetDim(0));  // M
    c->SetDim(1, sb.GetDim(1));  // N
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // 输入 fp16，输出 fp32：cube 内部按 fp32 累加
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

// ---- 角色 1：算子原型 -------------------------------------------------------
// 输入 a[M,K]、b[K,N]，输出 c[M,N]。注意 dtype 列表长度必须一致：
// 这里只声明一种组合（fp16 → fp32），想加 bf16 就要在每个列表里补一项。
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
        this->AICore().SetTiling(optiling::TilingMmadBasic);
        for (const auto *config : SGL_CANN_CONFIGS) {
            this->AICore().AddConfig(config);
        }
    }
};

OP_ADD(SglTeaching);

} // namespace ops
