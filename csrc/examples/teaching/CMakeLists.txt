# 教学算子声明。
#
# 这里不是"代码"，是给两套构建各喂一份元数据：
#   1) setup.py 的 stage③（build_cann）：用 OPDEF/INFER/TILING/KERNEL 编出真正的
#      CANN 算子包（含 aclnn 接口），装进 vendor；
#   2) setup.py 的 stage④（torch 扩展）：只取 TORCH_ADAPTER，编进
#      libsgl_kernel_npu.so，于是 Python 侧才有 torch.ops.npu.*。
#
# 所有教学例子共用这一个算子 ID（teaching）：CANN / torch 两级构建缓存都以
# --ops 的选择串为 key，多注册一个 ID 就必须 --make-clean。换例子时只改下面
# 四个源文件路径，选择串保持 "apply_token_bitmask,teaching" 不变。
#
# 为什么源码必须放在 csrc/ 下（而不是仓库根的 teaching/）：
#   csrc/cmake/cann_ops.cmake:198 用 ${CMAKE_CURRENT_SOURCE_DIR}（= csrc）当基准算
#   kernel 目录的相对路径，随后 CANN 的 ascendc_copy_kernel_src.py 要把 kernel
#   源码拷进 autogen/，它明确拒绝带 ".." 的路径
#   （报错 Unsafe relative path: [../teaching/...]）。
#   所以 kernel 源码必须位于 csrc 子树内；本例整棵树也就一起搬进来了。
#
# 另外不要用 ${PROJECT_SOURCE_DIR} 拼路径 —— 两个构建各自 project() 一次，根目录
# 已经被改写成 <repo>/csrc 和 <repo>/csrc/aclnn_torch_adapter。
sgl_register_operator(
    ID teaching
    EXECUTION_UNIT AICORE
    CANN_OP_TYPE SglTeaching
    TORCH_ADAPTER ${CMAKE_CURRENT_LIST_DIR}/01_mmad_basic/op_host/torch_adapter.cpp
    SUPPORTED_ARCHS 2201
    OPDEF_SOURCES ${CMAKE_CURRENT_LIST_DIR}/01_mmad_basic/op_host/mmad_basic_op.cpp
    INFER_SOURCES ${CMAKE_CURRENT_LIST_DIR}/01_mmad_basic/op_host/mmad_basic_op.cpp
    TILING_SOURCES ${CMAKE_CURRENT_LIST_DIR}/01_mmad_basic/op_host/mmad_basic_op.cpp
    KERNEL_SOURCES ${CMAKE_CURRENT_LIST_DIR}/01_mmad_basic/op_kernel/mmad_basic_kernel.cpp
)
