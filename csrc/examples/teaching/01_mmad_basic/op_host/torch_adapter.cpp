// 教学例子 01 的 torch 侧适配层：torch.ops.npu.<name>  ->  aclnn  ->  CANN 算子
//
// 这一层不做任何计算，只做三件事：校验、分配输出、把张量转成 aclnn 描述符并下发。
// 它和 sparse_flash_mla 上游那个 "OpBuilder + aclnnSparseFlashMla" 的位置完全一样，
// 只是那边是 JIT 编的 shim，我们这里是预先编进 libsgl_kernel_npu.so。

#include "aclnn_runtime.h"
#include "aclnn_sgl_teaching.h" // 由 opdef 自动生成，装在 vendor 的 op_api/include

#include <c10/core/DeviceGuard.h>
#include <limits>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

namespace
{

at::Tensor mmad_basic(const at::Tensor &a, const at::Tensor &b)
{
    TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1, "a must be on NPU");
    TORCH_CHECK(b.device() == a.device(), "inputs must be on the same NPU");
    TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf, "inputs must be fp16");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "inputs must be 2D");
    TORCH_CHECK(a.size(1) == b.size(1), "K mismatch: a is [M,K], b is [N,K]");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous");

    const c10::DeviceGuard guard(a.device());
    // C = A·Bᵀ -> [M, N] = [A 的行数, B 的行数]
    at::Tensor c = at_npu::native::empty_with_format(
        {a.size(0), b.size(0)}, a.options().dtype(at::kFloat), ACL_FORMAT_ND, true);

    static auto query = reinterpret_cast<decltype(&aclnnSglTeachingGetWorkspaceSize)>(
        sgl_cann::op_api_symbol("aclnnSglTeachingGetWorkspaceSize"));
    static auto execute = reinterpret_cast<decltype(&aclnnSglTeaching)>(sgl_cann::op_api_symbol("aclnnSglTeaching"));

    auto aDesc = sgl_cann::tensor_descriptor(a, ACL_FLOAT16);
    auto bDesc = sgl_cann::tensor_descriptor(b, ACL_FLOAT16);
    auto cDesc = sgl_cann::tensor_descriptor(c, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto status = query(aDesc.get(), bDesc.get(), cDesc.get(), &workspaceSize, &executor);
    TORCH_CHECK(status == ACL_SUCCESS, "MmadBasic workspace query failed: ", aclGetRecentErrMsg());
    TORCH_CHECK(workspaceSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "ACLNN workspace exceeds the Tensor allocation limit");

    auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, a.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    a.record_stream(stream);
    b.record_stream(stream);
    c.record_stream(stream);
    workspace.record_stream(stream);
    auto aclStream = stream.stream(false);
    // 捕获保护 host 队列，record_stream 保护设备侧异步使用
    auto launch = [a, b, c, workspace, aDesc, bDesc, cDesc, workspaceSize, executor, aclStream]() -> int {
        auto result = execute(workspaceSize ? workspace.data_ptr() : nullptr, workspaceSize, executor, aclStream);
        TORCH_CHECK(result == ACL_SUCCESS, "MmadBasic ACLNN execution failed: ", aclGetRecentErrMsg());
        return result;
    };
    at_npu::native::OpCommand::RunOpApi("aclnnSglTeaching", launch);
    return c;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("mmad_basic(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("mmad_basic", &mmad_basic);
}
