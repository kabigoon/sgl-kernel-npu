/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef SGL_KERNEL_NPU_KV_QUANT_SPARSE_ATTN_SHAREDKV_TILING_DATA_H
#define SGL_KERNEL_NPU_KV_QUANT_SPARSE_ATTN_SHAREDKV_TILING_DATA_H

#include <cstddef>
#include <cstdint>

namespace optiling {

struct KvQuantSparseAttnSharedkvBaseParams {
    uint32_t batchSize = 0;
    uint32_t qSeqSize = 0;
    uint32_t kvSeqSize = 0;
    uint32_t paOriBlockSize = 0;
    uint32_t paCmpBlockSize = 0;
    uint32_t oriMaxBlockNumPerBatch = 0;
    uint32_t cmpMaxBlockNumPerBatch = 0;
    uint32_t nNumOfQInOneGroup = 0;
    uint32_t sparseBlockCount = 0;
    float softmaxScale = 1.0F;
    int32_t oriKvStride = 0;
    int32_t cmpKvStride = 0;
    uint32_t tileSize = 64;
    uint32_t ropeHeadDim = 64;
    uint32_t cmpRatio = 1;
    uint32_t oriMaskMode = 4;
    uint32_t cmpMaskMode = 3;
    int32_t oriWinLeft = 127;
    int32_t oriWinRight = 0;
    uint32_t sparseBlockSize = 0;
    bool hasOriSparseIndices = false;
    uint8_t reserved[3] = {0, 0, 0};
    uint32_t oriSparseIndexWidth = 0;
    uint32_t dSize = 0;
    uint32_t dSizeVInput = 0;
};

struct KvQuantSparseAttnSharedkvTilingData {
    KvQuantSparseAttnSharedkvBaseParams baseParams{};
    uint32_t dispatchKey = 0;
};

static_assert(sizeof(KvQuantSparseAttnSharedkvBaseParams) == 96,
              "direct-launch base tiling ABI must match the upstream field layout");
static_assert(offsetof(KvQuantSparseAttnSharedkvTilingData, dispatchKey) == 96,
              "dispatch key must follow the upstream base tiling payload");
static_assert(sizeof(KvQuantSparseAttnSharedkvTilingData) == 100,
              "unexpected direct-launch tiling ABI padding");

constexpr uint32_t MakeKvQuantSASDispatchKey(uint32_t qLayout, uint32_t mode, uint32_t splitG)
{
    return (qLayout << 16U) | (mode << 8U) | splitG;
}

}  // namespace optiling
#endif
