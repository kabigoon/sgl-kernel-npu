# KV-quant sparse shared-KV metadata host scheduler

Source: `vllm-ascend/csrc/attention/kv_quant_sparse_attn_sharedkv_metadata` at
commit `b0c49dc7a884d3d343f6a556380f11b74191c3e8` (including DSpark A5 commit
`683b6c009`). The original source is licensed under CANN Open Software License
2.0; the license header is retained in the host implementation.

The upstream AICPU scheduler was reimplemented as an independent ordinary C++
host scheduler. `CpuKernelContext`, AICPU tensors, attributes, and registration
were removed. The Ascend 950 scheduling semantics are preserved, including
64/128 query-head handling, the 9-word FA metadata record,
`maxS2GBaseNum`, and the updated FD workspace accounting.

Sequence-length inputs are contiguous CPU int32 mirrors. The 4 KiB
`int32[1024]` result is produced in pinned memory and copied asynchronously to
the current NPU on the current torch-npu stream. Callers must retain/update CPU
mirrors instead of passing the external AICPU path's NPU seq-lens tensors.

The host scheduler is a control-plane operation and should run before NPU Graph
capture. Copy its returned table into the caller's persistent metadata buffer;
the captured attention launch then reuses that stable device address on replay.
