/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../op_host/kv_quant_sparse_attn_sharedkv_tiling_data.h"
#include "arch35/kv_quant_sparse_attn_sharedkv_scfa_kernel.h"

using namespace AscendC;

namespace {
constexpr int kLayoutBsnd = 0;
constexpr int kLayoutTnd = 1;
constexpr int kLayoutPaNd = 2;
constexpr int kSwaTemplate = 0;
constexpr int kCfaTemplate = 1;
constexpr int kScfaTemplate = 2;
}  // namespace

template<int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int SPLIT_G>
__aicore__ inline void LaunchKvQuantSparseAttnSharedkv(
    GM_ADDR query, GM_ADDR oriKV, GM_ADDR cmpKV, GM_ADDR oriSparseIndices, GM_ADDR cmpSparseIndices,
    GM_ADDR oriBlockTable, GM_ADDR cmpBlockTable, GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensOriKv,
    GM_ADDR cuSeqlensCmpKv, GM_ADDR seqUsedQ, GM_ADDR seqUsedKV, GM_ADDR sinks, GM_ADDR metadata,
    GM_ADDR attentionOut, GM_ADDR softmaxLse, GM_ADDR user,
    const optiling::KvQuantSparseAttnSharedkvTilingData *tilingData, TPipe *pipe)
{
    using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,
        BaseApi::SCFABlockCube<bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t, FLASH_DECODE, true,
            static_cast<SAS_LAYOUT>(LAYOUT_T), static_cast<SAS_LAYOUT>(KV_LAYOUT_T),
            static_cast<SASTemplateMode>(TEMPLATE_MODE), SPLIT_G>,
        BaseApi::SCFABlockCubeDummy<bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t, FLASH_DECODE, true,
            static_cast<SAS_LAYOUT>(LAYOUT_T), static_cast<SAS_LAYOUT>(KV_LAYOUT_T),
            static_cast<SASTemplateMode>(TEMPLATE_MODE), SPLIT_G>>::type;
    using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,
        BaseApi::SCFABlockVecDummy<bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t, FLASH_DECODE, true,
            static_cast<SAS_LAYOUT>(LAYOUT_T), static_cast<SAS_LAYOUT>(KV_LAYOUT_T),
            static_cast<SASTemplateMode>(TEMPLATE_MODE), SPLIT_G>,
        BaseApi::SCFABlockVec<bfloat16_t, fp8_e4m3fn_t, float, bfloat16_t, FLASH_DECODE, true,
            static_cast<SAS_LAYOUT>(LAYOUT_T), static_cast<SAS_LAYOUT>(KV_LAYOUT_T),
            static_cast<SASTemplateMode>(TEMPLATE_MODE), SPLIT_G>>::type;
    BaseApi::KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType> op;
    op.Init(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,
            cuSeqlensQ, seqUsedQ, seqUsedKV, sinks, metadata, attentionOut, user, tilingData, pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void kv_quant_sparse_attn_sharedkv(
    GM_ADDR query, GM_ADDR oriKV, GM_ADDR cmpKV, GM_ADDR oriSparseIndices, GM_ADDR cmpSparseIndices,
    GM_ADDR oriBlockTable, GM_ADDR cmpBlockTable, GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensOriKv,
    GM_ADDR cuSeqlensCmpKv, GM_ADDR seqUsedQ, GM_ADDR seqUsedKV, GM_ADDR sinks, GM_ADDR metadata,
    GM_ADDR attentionOut, GM_ADDR softmaxLse, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    // arch35 rejects passing an address-space-gm struct pointer to an API that
    // expects Local Memory. Keep the direct-launch ABI in GM, then copy scalar
    // fields explicitly before invoking the upstream template implementation.
    const __gm__ optiling::KvQuantSparseAttnSharedkvTilingData *__restrict__ tilingGm =
        reinterpret_cast<const __gm__ optiling::KvQuantSparseAttnSharedkvTilingData *>(tiling);
    optiling::KvQuantSparseAttnSharedkvTilingData tilingDataIn{};
    tilingDataIn.baseParams.batchSize = tilingGm->baseParams.batchSize;
    tilingDataIn.baseParams.qSeqSize = tilingGm->baseParams.qSeqSize;
    tilingDataIn.baseParams.kvSeqSize = tilingGm->baseParams.kvSeqSize;
    tilingDataIn.baseParams.paOriBlockSize = tilingGm->baseParams.paOriBlockSize;
    tilingDataIn.baseParams.paCmpBlockSize = tilingGm->baseParams.paCmpBlockSize;
    tilingDataIn.baseParams.oriMaxBlockNumPerBatch = tilingGm->baseParams.oriMaxBlockNumPerBatch;
    tilingDataIn.baseParams.cmpMaxBlockNumPerBatch = tilingGm->baseParams.cmpMaxBlockNumPerBatch;
    tilingDataIn.baseParams.nNumOfQInOneGroup = tilingGm->baseParams.nNumOfQInOneGroup;
    tilingDataIn.baseParams.sparseBlockCount = tilingGm->baseParams.sparseBlockCount;
    tilingDataIn.baseParams.softmaxScale = tilingGm->baseParams.softmaxScale;
    tilingDataIn.baseParams.oriKvStride = tilingGm->baseParams.oriKvStride;
    tilingDataIn.baseParams.cmpKvStride = tilingGm->baseParams.cmpKvStride;
    tilingDataIn.baseParams.tileSize = tilingGm->baseParams.tileSize;
    tilingDataIn.baseParams.ropeHeadDim = tilingGm->baseParams.ropeHeadDim;
    tilingDataIn.baseParams.cmpRatio = tilingGm->baseParams.cmpRatio;
    tilingDataIn.baseParams.oriMaskMode = tilingGm->baseParams.oriMaskMode;
    tilingDataIn.baseParams.cmpMaskMode = tilingGm->baseParams.cmpMaskMode;
    tilingDataIn.baseParams.oriWinLeft = tilingGm->baseParams.oriWinLeft;
    tilingDataIn.baseParams.oriWinRight = tilingGm->baseParams.oriWinRight;
    tilingDataIn.baseParams.sparseBlockSize = tilingGm->baseParams.sparseBlockSize;
    tilingDataIn.baseParams.hasOriSparseIndices = tilingGm->baseParams.hasOriSparseIndices;
    tilingDataIn.baseParams.oriSparseIndexWidth = tilingGm->baseParams.oriSparseIndexWidth;
    tilingDataIn.baseParams.dSize = tilingGm->baseParams.dSize;
    tilingDataIn.baseParams.dSizeVInput = tilingGm->baseParams.dSizeVInput;
    tilingDataIn.dispatchKey = tilingGm->dispatchKey;
    const optiling::KvQuantSparseAttnSharedkvTilingData *__restrict__ tilingData = &tilingDataIn;

#define KVQ_SAS_LAUNCH(layout, mode, split_g)                                                            \
    LaunchKvQuantSparseAttnSharedkv<0, layout, kLayoutPaNd, mode, split_g>(                               \
        query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,            \
        cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedQ, seqUsedKV, sinks, metadata,                 \
        attentionOut, softmaxLse, user, tilingData, &pipe)

    switch (tilingData->dispatchKey) {
        case optiling::MakeKvQuantSASDispatchKey(kLayoutBsnd, kSwaTemplate, 0):
            KVQ_SAS_LAUNCH(kLayoutBsnd, kSwaTemplate, 0); break;
        case optiling::MakeKvQuantSASDispatchKey(kLayoutBsnd, kSwaTemplate, 1):
            KVQ_SAS_LAUNCH(kLayoutBsnd, kSwaTemplate, 1); break;
        case optiling::MakeKvQuantSASDispatchKey(kLayoutBsnd, kCfaTemplate, 0):
            KVQ_SAS_LAUNCH(kLayoutBsnd, kCfaTemplate, 0); break;
        case optiling::MakeKvQuantSASDispatchKey(kLayoutBsnd, kCfaTemplate, 1):
            KVQ_SAS_LAUNCH(kLayoutBsnd, kCfaTemplate, 1); break;
        case optiling::MakeKvQuantSASDispatchKey(kLayoutBsnd, kScfaTemplate, 0):
            KVQ_SAS_LAUNCH(kLayoutBsnd, kScfaTemplate, 0); break;
        case optiling::MakeKvQuantSASDispatchKey(kLayoutBsnd, kScfaTemplate, 1):
            KVQ_SAS_LAUNCH(kLayoutBsnd, kScfaTemplate, 1); break;
        case optiling::MakeKvQuantSASDispatchKey(kLayoutTnd, kSwaTemplate, 0):
            KVQ_SAS_LAUNCH(kLayoutTnd, kSwaTemplate, 0); break;
        case optiling::MakeKvQuantSASDispatchKey(kLayoutTnd, kSwaTemplate, 1):
            KVQ_SAS_LAUNCH(kLayoutTnd, kSwaTemplate, 1); break;
        case optiling::MakeKvQuantSASDispatchKey(kLayoutTnd, kCfaTemplate, 0):
            KVQ_SAS_LAUNCH(kLayoutTnd, kCfaTemplate, 0); break;
        case optiling::MakeKvQuantSASDispatchKey(kLayoutTnd, kCfaTemplate, 1):
            KVQ_SAS_LAUNCH(kLayoutTnd, kCfaTemplate, 1); break;
        case optiling::MakeKvQuantSASDispatchKey(kLayoutTnd, kScfaTemplate, 0):
            KVQ_SAS_LAUNCH(kLayoutTnd, kScfaTemplate, 0); break;
        case optiling::MakeKvQuantSASDispatchKey(kLayoutTnd, kScfaTemplate, 1):
            KVQ_SAS_LAUNCH(kLayoutTnd, kScfaTemplate, 1); break;
    }
#undef KVQ_SAS_LAUNCH
}
