"""Static integration guards that run without an Ascend device."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _read(path):
    return (ROOT / path).read_text()


def test_direct_launch_has_no_opp_or_aclnn_dependency():
    host = _read("csrc/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv.cpp")
    kernel = _read("csrc/kv_quant_sparse_attn_sharedkv/op_kernel/kv_quant_sparse_attn_sharedkv.cpp")
    assert "EXEC_KERNEL_CMD(kv_quant_sparse_attn_sharedkv" in host
    assert 'extern "C" __global__ __aicore__ void kv_quant_sparse_attn_sharedkv' in kernel
    for forbidden in ("aclnnKvQuantSparseAttnSharedkv", "vllm_ascend_C", "libcust_opapi"):
        assert forbidden not in host + kernel


def test_launch_abi_optional_order():
    host = _read("csrc/kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv.cpp")
    kernel = _read("csrc/kv_quant_sparse_attn_sharedkv/op_kernel/kv_quant_sparse_attn_sharedkv.cpp")
    ordered = ["oriSparseIndices", "cmpSparseIndices", "oriBlockTable", "cmpBlockTable",
               "cuSeqlensQ", "cuSeqlensOriKv", "cuSeqlensCmpKv", "seqUsedQ", "seqUsedKV"]
    signature = kernel[kernel.index('extern "C"'): kernel.index(")\n{", kernel.index('extern "C"'))]
    assert [signature.index(name) for name in ordered] == sorted(signature.index(name) for name in ordered)
    assert "ori_sparse_indices.value_or(intPlaceholder)" in host


def test_arch35_tiling_is_copied_from_gm_before_template_dispatch():
    kernel = _read("csrc/kv_quant_sparse_attn_sharedkv/op_kernel/kv_quant_sparse_attn_sharedkv.cpp")
    assert "const __gm__ optiling::KvQuantSparseAttnSharedkvTilingData" in kernel
    assert "KvQuantSparseAttnSharedkvTilingData tilingDataIn{}" in kernel
    assert "tilingDataIn.baseParams.hasOriSparseIndices" in kernel
    assert "tilingDataIn.dispatchKey = tilingGm->dispatchKey" in kernel


def test_a5_gating_and_special_compile_options():
    cmake = _read("csrc/CMakeLists.txt")
    target = cmake[cmake.index("set(A5_ONLY_KERNEL_TARGETS)"): cmake.index("# create shared library")]
    assert "if(SGL_KERNEL_ENABLE_A5_ONLY_OPS)" in target
    for option in ("--cce-auto-sync=off", "-Wno-deprecated-declarations",
                   "-cce-vf-remove-membar=false", "-cce-aicore-hoist-movemask=false"):
        assert option in target


def test_metadata_is_independent_and_uses_a5_layout():
    metadata = _read("csrc/kv_quant_sparse_attn_sharedkv_metadata/op_host/kv_quant_sparse_attn_sharedkv_metadata.cpp")
    header = _read("csrc/kv_quant_sparse_attn_sharedkv_metadata/op_host/kv_quant_sparse_attn_sharedkv_metadata.h")
    assert "sgl_kernel_npu::SparseAttnSharedkvMetadataHost" not in metadata
    assert "FA_METADATA_SIZE = 9" in header
    assert "FA_S2_MAX_NUM = 8" in header
    assert "maxS2GBaseNum" in metadata and "isN128_" in metadata
    assert "pinned_memory(true)" in metadata and "non_blocking=*/true" in metadata
