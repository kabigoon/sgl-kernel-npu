// 教学例子 01 的 host 侧
//
// 只做三件事：
//   1. 检查形状/dtype/device（教学例子，检查够用就行）
//   2. 分配输出张量
//   3. EXEC_KERNEL_CMD 把 kernel 拉起来
//
// 这里没有 aclnn、没有 tiling、没有 workspace —— 因为 kernel 里的尺寸是写死的。
// 真实算子（sparse_flash_mla）多出来的那一大坨就是在解决"尺寸不能写死"。

#include "torch_helper.h"

#include "aclrtlaunch_mmad_basic.h"

#include <c10/core/DeviceGuard.h>

namespace
{

constexpr int64_t TILE = 16;

at::Tensor mmad_basic_launch(const at::Tensor &a, const at::Tensor &b)
{
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1, "a must be on NPU");
    TORCH_CHECK(b.device().type() == c10::DeviceType::PrivateUse1, "b must be on NPU");
    TORCH_CHECK(a.scalar_type() == at::kHalf, "a must be fp16");
    TORCH_CHECK(b.scalar_type() == at::kHalf, "b must be fp16");
    TORCH_CHECK(a.dim() == 2 && a.size(0) == TILE && a.size(1) == TILE, "a must be [16, 16]");
    TORCH_CHECK(b.dim() == 2 && b.size(0) == TILE && b.size(1) == TILE, "b must be [16, 16]");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");

    const c10::DeviceGuard guard(a.device());
    // 输出是 fp32：cube 内部按 fp32 累加，这正是我们要观察的点之一
    at::Tensor c = at::empty({TILE, TILE}, a.options().dtype(at::kFloat));

    // blockDim = 1：只用 1 个 AI Core 的一条 Mmad
    EXEC_KERNEL_CMD(mmad_basic, 1, a, b, c);
    return c;
}

} // namespace

// 和仓库其它算子一样，注册进 torch 的 npu 命名空间，Python 侧是 torch.ops.npu.mmad_basic
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("mmad_basic(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("mmad_basic", TORCH_FN(mmad_basic_launch));
}
