# KV-quant sparse shared-KV attention (Ascend 950)

This directory is derived from
`vllm-ascend/csrc/attention/kv_quant_sparse_attn_sharedkv` at source commit
`b0c49dc7a884d3d343f6a556380f11b74191c3e8`. That revision contains the
DSpark A5 update introduced by `683b6c009`, including `ori_sparse_indices`.

The upstream Ascend C device algorithm and its CANN Open Software License 2.0
headers are retained. The integration was adapted from an OPP/ACLNN operator to
sgl-kernel-npu's direct-launch architecture:

- a stable non-template `extern "C"` kernel entry dispatches the upstream
  template matrix from a cached dispatch key;
- PyTorch host code performs validation, invokes the upstream checker/tiler
  through the local `ge_helper` context, caches tiling tensors for NPU Graph,
  allocates workspace, and calls `EXEC_KERNEL_CMD`;
- the kernel is compiled only for Ascend 950 in a dedicated target with the
  upstream CCE compiler workarounds.

It does not call `aclnnKvQuantSparseAttnSharedkv`, load `vllm_ascend_C.so`, or
install/register a `custom_transformer` OPP package.
