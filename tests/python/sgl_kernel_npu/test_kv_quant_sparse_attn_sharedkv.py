"""Direct-launch tests for the Ascend 950 BF16-Q / FP8-KV sparse attention kernel."""

import math
import os
import subprocess

import pytest
import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu
from utils import require_npu_op

pytestmark = require_npu_op("kv_quant_sparse_attn_sharedkv")
OP = torch.ops.npu.kv_quant_sparse_attn_sharedkv


def _single_core_metadata(device, heads, batch, kv_len):
    meta = torch.zeros(1024, dtype=torch.int32, device=device)
    # Ascend 950 FA record: enable, BN2/M/S2 start, BN2/M/S2 end,
    # first FD workspace index, maximum S2 base blocks.
    record = torch.tensor(
        [1, 0, 0, 0, batch, 0, 0, 0, batch * math.ceil(kv_len / 128)],
        dtype=torch.int32,
        device=device,
    )
    meta[:9] = record
    # The 128-head A5 schedule pairs physical cube cores for one logical row.
    if heads == 128:
        meta[9:18] = record
    return meta


def _inputs(heads=64, batch=1, q_len=1, kv_len=128, sparse_ori=False, compressed=False, sparse_cmp=False):
    device, block = torch.device("npu:0"), 128
    pages = batch * ((kv_len + block - 1) // block + 1)
    q = torch.zeros((batch * q_len, heads, 512), dtype=torch.bfloat16, device=device)
    ori = torch.zeros((pages, block, 1, 640), dtype=torch.float8_e4m3fn, device=device)
    table = torch.arange(pages, dtype=torch.int32, device=device).view(batch, -1)
    cu = torch.arange(0, (batch + 1) * q_len, q_len, dtype=torch.int32, device=device)
    used = torch.full((batch,), kv_len, dtype=torch.int32, device=device)
    sinks = torch.full((heads,), -1e4, dtype=torch.float32, device=device)
    kwargs = dict(q=q, kv_quant_mode=1, ori_kv=ori, ori_block_table=table,
                  cu_seqlens_q=cu, seqused_kv=used, sinks=sinks,
                  metadata=_single_core_metadata(device, heads, batch, kv_len), tile_size=64, rope_head_dim=64,
                  softmax_scale=1 / math.sqrt(512), cmp_ratio=4 if compressed else 1,
                  ori_win_left=127, ori_win_right=0, layout_q="TND", layout_kv="PA_ND")
    if sparse_ori:
        # Absolute physical slot ids; intentionally non-monotonic to exercise 683b6c009.
        kwargs["ori_sparse_indices"] = torch.tensor([[[17, 3, 95, 1]]], dtype=torch.int32, device=device).expand(batch * q_len, 1, 4).contiguous()
    if compressed:
        kwargs["cmp_kv"] = torch.zeros_like(ori)
        kwargs["cmp_block_table"] = table.clone()
    if sparse_cmp:
        kwargs["cmp_sparse_indices"] = torch.zeros((batch * q_len, 1, 512), dtype=torch.int32, device=device)
    return kwargs


@pytest.mark.parametrize("heads,batch,kv_len", [(64, 1, 128), (128, 1, 256), (64, 3, 513)])
def test_zero_cpu_reference_and_noncontiguous_pages(heads, batch, kv_len):
    kwargs = _inputs(heads, batch, 1, kv_len)
    # Reverse the physical mapping while preserving the logical sequence.
    kwargs["ori_block_table"] = kwargs["ori_block_table"].flip(1).contiguous()
    out, lse = OP(**kwargs)
    torch_npu.npu.synchronize()
    torch.testing.assert_close(out.cpu().float(), torch.zeros_like(out.cpu().float()), rtol=0, atol=0)
    assert lse.dtype == torch.float32 and lse.numel() == 0


def test_ori_sparse_indices_swa_core_scenario():
    out, _ = OP(**_inputs(sparse_ori=True))
    torch_npu.npu.synchronize()
    assert torch.count_nonzero(out).item() == 0


@pytest.mark.parametrize("sparse_cmp", [False, True])
def test_compressed_and_sparse_compressed_kv(sparse_cmp):
    out, _ = OP(**_inputs(compressed=True, sparse_cmp=sparse_cmp))
    torch_npu.npu.synchronize()
    assert out.shape[-1] == 512


def test_optional_and_error_checks():
    kwargs = _inputs()
    kwargs.pop("metadata")
    with pytest.raises(RuntimeError, match="metadata"):
        OP(**kwargs)
    kwargs = _inputs()
    kwargs["ori_kv"] = kwargs["ori_kv"].to(torch.float8_e5m2)
    with pytest.raises(RuntimeError, match="e4m3"):
        OP(**kwargs)


def test_graph_warmup_capture_and_replay():
    kwargs = _inputs(sparse_ori=True)
    OP(**kwargs)  # populate stable tiling cache outside capture
    torch_npu.npu.synchronize()
    graph = torch_npu.npu.NPUGraph()
    stream = torch_npu.npu.Stream()
    with torch_npu.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        captured, _ = OP(**kwargs)
    for _ in range(3):
        graph.replay()
    torch_npu.npu.synchronize()
    assert captured.shape == kwargs["q"].shape


def test_end_to_end_metadata_host_then_kernel():
    kwargs = _inputs(batch=2)
    cpu_cu = kwargs["cu_seqlens_q"].cpu()
    cpu_used = kwargs["seqused_kv"].cpu()
    kwargs["metadata"] = torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host(
        64, 1, 512, 1, cpu_cu, None, None, None, cpu_used, 2, 1, 128,
        0, 0, 64, 64, 1, 4, 3, 127, 0, "TND", "PA_ND", True, False)
    out, _ = OP(**kwargs)
    torch_npu.npu.synchronize()
    assert out.shape == kwargs["q"].shape


def test_registration_is_independent_of_vllm_ascend():
    assert hasattr(torch.ops.npu, "kv_quant_sparse_attn_sharedkv")
    assert hasattr(torch.ops.npu, "kv_quant_sparse_attn_sharedkv_metadata_host")
    assert hasattr(torch.ops.npu, "compressor")
    assert "_C_ascend" not in str(OP.default._schema)


@pytest.mark.skipif(not os.getenv("VLLM_ASCEND_PYTHON"), reason="set VLLM_ASCEND_PYTHON for isolated upstream comparison")
def test_matches_original_vllm_ascend_in_separate_process(tmp_path):
    expected = tmp_path / "upstream_attention.pt"
    code = f"""
import math, torch, torch_npu, vllm_ascend
d='npu:0'; q=torch.zeros((1,64,512),dtype=torch.bfloat16,device=d)
kv=torch.zeros((2,128,1,640),dtype=torch.float8_e4m3fn,device=d)
bt=torch.tensor([[1,0]],dtype=torch.int32,device=d); cu=torch.tensor([0,1],dtype=torch.int32,device=d)
used=torch.tensor([128],dtype=torch.int32,device=d); sinks=torch.full((64,),-1e4,dtype=torch.float32,device=d)
meta=torch.zeros(1024,dtype=torch.int32,device=d); meta[:9]=torch.tensor([1,0,0,0,1,0,0,0,1],dtype=torch.int32,device=d)
o,_=torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv(q,1,kv,None,None,None,bt,None,cu,None,None,None,used,sinks,meta,64,64,1/math.sqrt(512),1,4,3,127,0,'TND','PA_ND',False)
torch.save(o.cpu(),r'{expected}')
"""
    subprocess.run([os.environ["VLLM_ASCEND_PYTHON"], "-c", code], check=True)
    actual, _ = OP(**_inputs())
    torch_npu.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), torch.load(expected, weights_only=True), rtol=0, atol=0)
