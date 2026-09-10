/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <limits>
#include <memory>
#include <string>

#include "aclrtlaunch_kv_quant_sparse_attn_sharedkv.h"
#include "ge_helper.h"
#include "kv_quant_sparse_attn_sharedkv_def.h"
#include "kv_quant_sparse_attn_sharedkv_tiling.h"
#include "torch_helper.h"

namespace sglang::npu_kernel {
namespace {
using ge_helper::TilingContext;

void CheckNpuTensor(const at::Tensor &tensor, const at::Tensor &q, const char *name)
{
    TORCH_CHECK(tensor.device() == q.device(), name, " must be on the same NPU device as q");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckKvTensor(const at::Tensor &tensor, const at::Tensor &q, const char *name)
{
    TORCH_CHECK(tensor.device() == q.device(), name, " must be on the same NPU device as q");
    TORCH_CHECK(tensor.dim() == 4, name, " must have PA_ND rank 4");
    int64_t expectedStride = 1;
    for (int64_t axis = tensor.dim() - 1; axis >= 1; --axis) {
        TORCH_CHECK(tensor.stride(axis) == expectedStride, name,
                    " must be contiguous below dimension 0; got stride(", axis, ")=", tensor.stride(axis),
                    ", expected ", expectedStride);
        expectedStride *= tensor.size(axis);
    }
    TORCH_CHECK(tensor.stride(0) >= expectedStride, name,
                " stride(0) must cover at least one physical page; got ", tensor.stride(0),
                ", expected >= ", expectedStride);
}

void CheckOptional(const c10::optional<at::Tensor> &tensor, const at::Tensor &q, const char *name)
{
    if (tensor.has_value()) CheckNpuTensor(*tensor, q, name);
}

at::Tensor Placeholder(const at::Tensor &q, at::ScalarType dtype)
{
    return at::empty({1}, q.options().dtype(dtype));
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> kv_quant_sparse_attn_sharedkv(
    const at::Tensor &q, int64_t kv_quant_mode, const c10::optional<at::Tensor> &ori_kv,
    const c10::optional<at::Tensor> &cmp_kv, const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &cmp_sparse_indices, const c10::optional<at::Tensor> &ori_block_table,
    const c10::optional<at::Tensor> &cmp_block_table, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata,
    int64_t tile_size, int64_t rope_head_dim, double softmax_scale, int64_t cmp_ratio,
    int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left, int64_t ori_win_right,
    c10::string_view layout_q, c10::string_view layout_kv, bool return_softmax_lse)
{
    constexpr const char *op = "kv_quant_sparse_attn_sharedkv";
    TORCH_CHECK(q.device().type() == DEVICE_TYPE && q.scalar_type() == at::kBFloat16,
                op, ": q must be a BF16 NPU tensor");
    TORCH_CHECK(q.is_contiguous(), op, ": q must be contiguous");
    TORCH_CHECK((layout_q == "TND" && q.dim() == 3) || (layout_q == "BSND" && q.dim() == 4),
                op, ": q rank does not match layout_q");
    TORCH_CHECK(layout_kv == "PA_ND", op, ": only layout_kv='PA_ND' is supported on A5");
    TORCH_CHECK(ori_kv.has_value(), op, ": ori_kv is required");
    TORCH_CHECK(ori_kv->scalar_type() == at::kFloat8_e4m3fn, op, ": ori_kv must be float8_e4m3fn");
    TORCH_CHECK(!cmp_kv || cmp_kv->scalar_type() == at::kFloat8_e4m3fn, op, ": cmp_kv must be float8_e4m3fn");
    TORCH_CHECK(ori_block_table.has_value() && seqused_kv.has_value(),
                op, ": ori_block_table and seqused_kv are required for PA_ND");
    TORCH_CHECK(!cmp_kv || cmp_block_table, op, ": cmp_block_table is required with cmp_kv");
    TORCH_CHECK(!cmp_sparse_indices || cmp_kv, op, ": cmp_kv is required with cmp_sparse_indices");
    TORCH_CHECK(layout_q != "TND" || cu_seqlens_q, op, ": cu_seqlens_q is required for TND");
    TORCH_CHECK(metadata && metadata->scalar_type() == at::kInt && metadata->dim() == 1 && metadata->numel() == 1024,
                op, ": metadata must be NPU int32[1024]");
    TORCH_CHECK(sinks && sinks->scalar_type() == at::kFloat, op, ": sinks must be float32");
    TORCH_CHECK(!return_softmax_lse, op, ": return_softmax_lse=true is not supported by the A5 kernel");
    TORCH_CHECK(kv_quant_mode == 1, op, ": kv_quant_mode must be 1 (FP8 E4M3)");
    TORCH_CHECK(tile_size == 64 && rope_head_dim == 64, op, ": tile_size and rope_head_dim must both be 64");
    TORCH_CHECK(layout_q != "BSND" || !cu_seqlens_q, op, ": cu_seqlens_q must be None for BSND");
    TORCH_CHECK(softmax_scale >= 0.0, op, ": softmax_scale must be non-negative");
    TORCH_CHECK(cmp_ratio >= 0 && cmp_ratio <= std::numeric_limits<uint32_t>::max() && ori_mask_mode >= 0 &&
                    ori_mask_mode <= std::numeric_limits<uint32_t>::max() && cmp_mask_mode >= 0 &&
                    cmp_mask_mode <= std::numeric_limits<uint32_t>::max() && ori_win_left >= 0 &&
                    ori_win_left <= std::numeric_limits<int32_t>::max() && ori_win_right >= 0 &&
                    ori_win_right <= std::numeric_limits<int32_t>::max() &&
                    ori_kv->stride(0) <= std::numeric_limits<int32_t>::max() &&
                    (!cmp_kv || cmp_kv->stride(0) <= std::numeric_limits<int32_t>::max()),
                op, ": integer attributes or KV stride exceed the tiling ABI range");

    auto checkInt32 = [](const c10::optional<at::Tensor> &t, const char *name) {
        TORCH_CHECK(!t || t->scalar_type() == at::kInt, name, " must be int32");
    };
    checkInt32(ori_sparse_indices, "ori_sparse_indices"); checkInt32(cmp_sparse_indices, "cmp_sparse_indices");
    checkInt32(ori_block_table, "ori_block_table"); checkInt32(cmp_block_table, "cmp_block_table");
    checkInt32(cu_seqlens_q, "cu_seqlens_q"); checkInt32(cu_seqlens_ori_kv, "cu_seqlens_ori_kv");
    checkInt32(cu_seqlens_cmp_kv, "cu_seqlens_cmp_kv"); checkInt32(seqused_q, "seqused_q");
    checkInt32(seqused_kv, "seqused_kv");
    if (ori_sparse_indices) {
        const int64_t expectedRank = layout_q == "TND" ? 3 : 4;
        TORCH_CHECK(ori_sparse_indices->dim() == expectedRank, op,
                    ": ori_sparse_indices rank must match layout_q");
        TORCH_CHECK(ori_sparse_indices->size(0) == q.size(0), op,
                    ": ori_sparse_indices leading dimension must match q");
        if (layout_q == "BSND") {
            TORCH_CHECK(ori_sparse_indices->size(1) == q.size(1), op,
                        ": ori_sparse_indices query length must match q");
        }
        TORCH_CHECK(ori_sparse_indices->size(expectedRank - 2) == 1 &&
                        ori_sparse_indices->size(expectedRank - 1) > 0,
                    op, ": ori_sparse_indices must have one KV head and a non-empty index width");
    }
    CheckKvTensor(*ori_kv, q, "ori_kv");
    if (cmp_kv) CheckKvTensor(*cmp_kv, q, "cmp_kv");
    CheckOptional(ori_sparse_indices, q, "ori_sparse_indices"); CheckOptional(cmp_sparse_indices, q, "cmp_sparse_indices");
    CheckOptional(ori_block_table, q, "ori_block_table"); CheckOptional(cmp_block_table, q, "cmp_block_table");
    CheckOptional(cu_seqlens_q, q, "cu_seqlens_q"); CheckOptional(cu_seqlens_ori_kv, q, "cu_seqlens_ori_kv");
    CheckOptional(cu_seqlens_cmp_kv, q, "cu_seqlens_cmp_kv"); CheckOptional(seqused_q, q, "seqused_q");
    CheckOptional(seqused_kv, q, "seqused_kv"); CheckNpuTensor(*sinks, q, "sinks");
    CheckNpuTensor(*metadata, q, "metadata");

    auto attention_out = at::empty_like(q);
    auto softmax_lse = at::empty({0}, q.options().dtype(at::kFloat));
    KvQuantSASHost::KvQuantSparseAttnSharedkv def(op);
    def.SetAttrAny("kv_quant_mode", kv_quant_mode); def.SetAttrAny("tile_size", tile_size);
    def.SetAttrAny("rope_head_dim", rope_head_dim); def.SetAttrAny("softmax_scale", static_cast<float>(softmax_scale));
    def.SetAttrAny("cmp_ratio", cmp_ratio); def.SetAttrAny("ori_mask_mode", static_cast<uint32_t>(ori_mask_mode));
    def.SetAttrAny("cmp_mask_mode", static_cast<uint32_t>(cmp_mask_mode)); def.SetAttrAny("ori_win_left", ori_win_left);
    def.SetAttrAny("ori_win_right", ori_win_right); def.SetAttrStr("layout_q", std::string(layout_q));
    def.SetAttrStr("layout_kv", std::string(layout_kv)); def.SetAttrAny("ori_kv_stride", ori_kv->stride(0));
    def.SetAttrAny("cmp_kv_stride", cmp_kv ? cmp_kv->stride(0) : int64_t{0});
    auto context = std::make_shared<TilingContext>(op);
    auto qType = q.scalar_type(); def.SetToContext(context, qType);
    context->RegisterTensor(q, true); context->RegisterTensor(ori_kv, true); context->RegisterTensor(cmp_kv, true);
    context->RegisterTensor(ori_sparse_indices, true); context->RegisterTensor(cmp_sparse_indices, true);
    context->RegisterTensor(ori_block_table, true); context->RegisterTensor(cmp_block_table, true);
    context->RegisterTensor(cu_seqlens_q, true); context->RegisterTensor(cu_seqlens_ori_kv, true);
    context->RegisterTensor(cu_seqlens_cmp_kv, true); context->RegisterTensor(seqused_q, true);
    context->RegisterTensor(seqused_kv, true); context->RegisterTensor(sinks, true);
    context->RegisterTensor(metadata, true); context->RegisterTensor(attention_out, false);
    context->RegisterTensor(softmax_lse, false);
    optiling::KvQuantSASTilingInfo info;
    optiling::KvQuantSASInfoParser parser(context.get());
    TORCH_CHECK(parser.Parse(info) == ge::GRAPH_SUCCESS, op, ": parsing tiling inputs failed");
    optiling::KvQuantSASTilingCheck checker(info);
    TORCH_CHECK(checker.Process() == ge::GRAPH_SUCCESS, op, ": tiling validation failed");
    optiling::KvQuantSparseAttnSharedkvTiling tiling(context.get());
    TORCH_CHECK(tiling.DoOpTiling(&info) == ge::GRAPH_SUCCESS, op, ": tiling failed");
    auto tilingTensor = context->GetTilingTensor(tiling.GetTilingData());
    auto workspace = at::empty({static_cast<int64_t>(context->GetWorkspaceSize())}, q.options().dtype(at::kByte));
    auto fp8Placeholder = Placeholder(q, at::kFloat8_e4m3fn);
    auto intPlaceholder = Placeholder(q, at::kInt);
    EXEC_KERNEL_CMD(kv_quant_sparse_attn_sharedkv, tiling.GetBlockDim(), q, *ori_kv,
        cmp_kv.value_or(fp8Placeholder), ori_sparse_indices.value_or(intPlaceholder),
        cmp_sparse_indices.value_or(intPlaceholder), ori_block_table.value_or(intPlaceholder),
        cmp_block_table.value_or(intPlaceholder), cu_seqlens_q.value_or(intPlaceholder),
        cu_seqlens_ori_kv.value_or(intPlaceholder), cu_seqlens_cmp_kv.value_or(intPlaceholder),
        seqused_q.value_or(intPlaceholder), seqused_kv.value_or(intPlaceholder), *sinks, *metadata,
        attention_out, softmax_lse, workspace, tilingTensor);
    return {attention_out, softmax_lse};
}
}  // namespace sglang::npu_kernel
