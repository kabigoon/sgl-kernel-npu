"""Ascend 950 tests for the independent KV-quant metadata host scheduler."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu
from utils import require_npu_op

pytestmark = require_npu_op("kv_quant_sparse_attn_sharedkv_metadata_host")
OP = torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata_host
FA_SLOTS, FA_WORDS, FD_SLOTS, FD_WORDS = 36, 9, 72, 8


def _call(heads=64, seq=(1,), kv=(512,), ratio=1, topk=0, has_ori=True, has_cmp=False):
    cu = torch.tensor((0, *torch.tensor(seq).cumsum(0).tolist()), dtype=torch.int32)
    seq_q = torch.tensor(seq, dtype=torch.int32)
    seq_kv = torch.tensor(kv, dtype=torch.int32)
    return OP(
        heads, 1, 512, 1, cu, None, None, seq_q, seq_kv, len(seq), max(seq),
        max(kv), 0, topk, 64, 64, ratio, 4, 3, 127, 0, "TND", "PA_ND",
        has_ori, has_cmp,
    )


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("seq,kv", [((1,), (128,)), ((1, 3, 7), (128, 1024, 8192))])
@pytest.mark.parametrize("ratio,topk,has_cmp", [(1, 0, False), (4, 0, True), (4, 512, True), (128, 1024, True)])
def test_contract_and_schedule_matrix(heads, seq, kv, ratio, topk, has_cmp):
    out = _call(heads, seq, kv, ratio, topk, True, has_cmp)
    assert out.dtype == torch.int32 and out.shape == (1024,) and out.device.type == "npu"
    host = out.cpu()
    fa = host[: FA_SLOTS * FA_WORDS].view(FA_SLOTS, FA_WORDS)
    assert int((fa[:, 0] == 1).sum()) >= 1
    if heads == 128:
        assert torch.equal(fa[0], fa[1])


@pytest.mark.parametrize("has_ori,has_cmp,ratio,topk", [(True, False, 1, 0), (False, True, 4, 0), (True, True, 4, 512)])
def test_kv_presence_combinations(has_ori, has_cmp, ratio, topk):
    _call(64, (1, 2), (256, 512), ratio, topk, has_ori, has_cmp)


@pytest.mark.parametrize("kwargs", [{"heads": 32}, {"ratio": 7, "topk": 512, "has_cmp": True}, {"ratio": 4, "topk": 64, "has_cmp": True}])
def test_invalid_parameters(kwargs):
    with pytest.raises(RuntimeError):
        _call(**kwargs)


def test_same_stream_consumption_and_staging_lifetime():
    stream = torch_npu.npu.Stream()
    with torch_npu.npu.stream(stream):
        metadata = _call(64, (1, 1), (1024, 2048))
        # The reduction is enqueued after the non-blocking H2D on the same stream.
        observed = metadata.sum()
    stream.synchronize()
    assert observed.cpu().item() >= 0


def test_npu_seq_lens_are_rejected_instead_of_implicit_d2h():
    cu = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    seq_kv = torch.tensor([128], dtype=torch.int32)
    with pytest.raises(RuntimeError):
        OP(64, 1, 512, 1, cu, None, None, None, seq_kv, 1, 1, 128,
           0, 0, 64, 64, 1, 4, 3, 127, 0, "TND", "PA_ND", True, False)


@pytest.mark.skipif(not os.getenv("VLLM_ASCEND_PYTHON"), reason="set VLLM_ASCEND_PYTHON for isolated AICPU comparison")
def test_byte_identical_to_vllm_ascend_aicpu(tmp_path):
    """Run the OPP/AICPU implementation in another process to avoid duplicate vendor registration."""
    expected = tmp_path / "aicpu.pt"
    code = f"""
import torch, torch_npu, vllm_ascend
cu=torch.tensor([0,1,4],dtype=torch.int32,device='npu')
skv=torch.tensor([1024,4096],dtype=torch.int32,device='npu')
o=torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata(64,1,512,1,cu,None,None,None,skv,2,3,4096,0,512,64,64,4,4,3,127,0,'TND','PA_ND',True,True,'npu')
torch.save(o.cpu(),r'{expected}')
"""
    subprocess.run([os.environ["VLLM_ASCEND_PYTHON"], "-c", code], check=True)
    actual = _call(64, (1, 3), (1024, 4096), 4, 512, True, True).cpu()
    assert torch.equal(actual, torch.load(expected, weights_only=True))


def test_schema_is_a5_host_variant():
    schema = str(OP.default._schema)
    assert "kv_quant_mode" in schema and "seqused_q" in schema
    assert "device" not in schema
