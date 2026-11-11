// host 与 kernel 共享的 tiling 结构。
//
// 例子 01 只需要一个字段（开几个核）；例子 02/03 会把它扩成
// {mBase, nBase, kBase, loopCount...}，那时就能对照 sparse_flash_mla 的
// SMLATilingData（60+ 字段）看出"tiling data = 给 device 传的一小包常量"。
#pragma once

#include <cstdint>

namespace optiling
{
struct SglTeachingTilingData {
    uint32_t blockDim;
};
static_assert(sizeof(SglTeachingTilingData) == sizeof(uint32_t),
              "Host and Kernel must share the same tiling layout");
} // namespace optiling
