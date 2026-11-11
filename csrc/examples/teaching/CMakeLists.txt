# 教学算子声明。
#
# 这里不是"代码"，是给两套构建各喂一份元数据：
#   1) setup.py 的 stage③（build_cann）：用 OPDEF/INFER/TILING/KERNEL 编出真正的
#      CANN 算子包（含 aclnn 接口），装进 vendor；
#   2) setup.py 的 stage④（torch 扩展）：只取 TORCH_ADAPTER，编进
#      libsgl_kernel_npu.so，于是 Python 侧才有 torch.ops.npu.*。
#
# 所有教学例子共用这一个算子 ID（teaching）：
# CANN 的构建缓存以 --ops 的选择串为 key，多注册一个 ID 就必须 --make-clean。
# 换例子时只改下面四个源文件路径，选择串保持 "apply_token_bitmask,teaching" 不变。
#
# 源文件放在 teaching/ 下，不在 csrc/ 里；这里只做"指路"。
sgl_register_operator(
    ID teaching
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglTeaching
    TORCH_ADAPTER ${PROJECT_SOURCE_DIR}/teaching/01_mmad_basic/op_host/torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES ${PROJECT_SOURCE_DIR}/teaching/01_mmad_basic/op_host/mmad_basic_op.cpp
    INFER_SOURCES ${PROJECT_SOURCE_DIR}/teaching/01_mmad_basic/op_host/mmad_basic_op.cpp
    TILING_SOURCES ${PROJECT_SOURCE_DIR}/teaching/01_mmad_basic/op_host/mmad_basic_op.cpp
    KERNEL_SOURCES ${PROJECT_SOURCE_DIR}/teaching/01_mmad_basic/op_kernel/mmad_basic_kernel.cpp
)
