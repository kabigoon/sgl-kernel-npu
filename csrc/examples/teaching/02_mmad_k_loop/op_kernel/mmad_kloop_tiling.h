// host 与 kernel 共享的 tiling 结构。
//
// 例子 01 只有 1 个字段（blockDim）；02 增加了两个，而且 kernel 真的会读它们：
//   kTotal = GM 里每行有多少个 K 元素（= 行间隔，喂给 Nd2NzParams.srcDValue）
//   kLoop  = 要循环几次 Mmad（= kTotal / 16）
//
// 这就是"参数/形状决定实现"的最小样本：同一个 kernel，K 变了只需要换 tiling data，
// 不用重新编译。真实算子（sparse_flash_mla）的 tiling data 有 48 个字段，作用相同。
#pragma once

#include <cstdint>

namespace optiling
{
struct SglTeachingTilingData {
    uint32_t blockDim;
    uint32_t kLoop;
    uint32_t kTotal;
};
static_assert(sizeof(SglTeachingTilingData) == 3 * sizeof(uint32_t),
              "Host and Kernel must share the same tiling layout");
} // namespace optiling
