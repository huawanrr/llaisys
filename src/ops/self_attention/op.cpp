#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(q->ndim() == 3, "SelfAttention: q must be 3D [seqlen, nhead, d].");
    CHECK_ARGUMENT(k->ndim() == 3, "SelfAttention: k must be 3D [total_len, nkvhead, d].");
    CHECK_ARGUMENT(v->ndim() == 3, "SelfAttention: v must be 3D [total_len, nkvhead, dv].");
    CHECK_ARGUMENT(attn_val->ndim() == 3, "SelfAttention: attn_val must be 3D [seqlen, nhead, dv].");

    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];

    CHECK_ARGUMENT(k->shape()[2] == d, "SelfAttention: k head dim must match q head dim.");
    CHECK_ARGUMENT(v->shape()[0] == total_len, "SelfAttention: v total_len must match k total_len.");
    CHECK_ARGUMENT(v->shape()[1] == nkvhead, "SelfAttention: v nkvhead must match k nkvhead.");
    CHECK_ARGUMENT(attn_val->shape()[0] == seqlen, "SelfAttention: attn_val seqlen must match q seqlen.");
    CHECK_ARGUMENT(attn_val->shape()[1] == nhead, "SelfAttention: attn_val nhead must match q nhead.");
    CHECK_ARGUMENT(attn_val->shape()[2] == dv, "SelfAttention: attn_val dv must match v dv.");
    CHECK_ARGUMENT(nhead % nkvhead == 0, "SelfAttention: nhead must be divisible by nkvhead.");
    CHECK_ARGUMENT(total_len >= seqlen, "SelfAttention: total_len must be >= seqlen.");

    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  seqlen, total_len, nhead, nkvhead, d, dv, scale, attn_val->dtype());
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  seqlen, total_len, nhead, nkvhead, d, dv, scale, attn_val->dtype());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
