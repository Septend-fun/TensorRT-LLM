"""Spark asymmetric INT4 prefill over TinyGEMM-packed weights."""

import torch
import triton
import triton.language as tl


@triton.jit
def _dequantize(w, sz, packed_offset, valid, qparam, qvalid,
                BN: tl.constexpr, BK: tl.constexpr):
    word = tl.load(w + packed_offset, valid, other=0).to(tl.uint32)
    nibble = tl.arange(0, 8)
    q = ((word[:, :, :, None] >> (nibble[None, None, None, :] * 4)) & 15).to(tl.int32) - 8
    scale = tl.load(sz + qparam, qvalid, other=0).to(tl.float32)
    zero = tl.load(sz + qparam + 1, qvalid, other=0).to(tl.float32)
    value = tl.fma(q.to(tl.float32), scale[:, :, None, None], zero[:, :, None, None])
    value = value.to(tl.bfloat16).reshape(BN, BK // 32, 4, 2, 4)
    return value.permute(0, 1, 4, 2, 3).reshape(BN, BK)


@triton.jit
def _gemm(x, w, sz, y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          GROUP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
          BK: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group_span = 8 * num_n
    first_m = (pid // group_span) * 8
    group_m = tl.minimum(num_m - first_m, 8)
    pm = first_m + (pid % group_span) % group_m
    pn = (pid % group_span) // group_m
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    groups = tl.arange(0, BK // 32)
    lanes = tl.arange(0, 4)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        ks = start * BK + rk
        a = tl.load(x + rows[:, None] * K + ks[None, :],
                    (rows[:, None] < M) & (ks[None, :] < K), other=0)
        kg = start * (BK // 32) + groups
        packed_offset = (((cols[:, None, None] // 8) * (K // 128) +
                          kg[None, :, None] // 4) * 128 +
                         (cols[:, None, None] % 8) * 16 +
                         lanes[None, None, :] * 4 + kg[None, :, None] % 4)
        valid = ((cols[:, None, None] < N) & (kg[None, :, None] * 32 < K))
        qparam = ((kg[None, :] * 32 // GROUP) * N + cols[:, None]) * 2
        qvalid = (cols[:, None] < N) & (kg[None, :] * 32 < K)
        acc = tl.dot(a, _dequantize(w, sz, packed_offset, valid, qparam,
                                    qvalid, BN, BK).T, acc)
    tl.store(y + rows[:, None] * N + cols[None, :], acc.to(tl.bfloat16),
             (rows[:, None] < M) & (cols[None, :] < N))


def linear(x: torch.Tensor, weight: torch.Tensor, scale_zero: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or x.ndim != 2 or not all(
            t.is_cuda and t.is_contiguous() for t in (x, weight, scale_zero)):
        raise ValueError("Spark W4 prefill requires contiguous CUDA BF16 tensors")
    if weight.dtype != torch.int32 or weight.ndim != 4 or scale_zero.dtype != torch.bfloat16:
        raise ValueError("invalid TinyGEMM packed Spark W4 tensors")
    m, k = x.shape
    n = weight.shape[0] * 8
    group = k // scale_zero.shape[0]
    if k % 128 or group not in (32, 128):
        raise ValueError("Spark W4 prefill supports G32/G128 and K divisible by 128")
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    _gemm[(triton.cdiv(m, 128) * triton.cdiv(n, 64),)](
        x, weight, scale_zero, output, m, n, k, group, 128, 64, 64,
        num_warps=4, num_stages=3)
    return output
