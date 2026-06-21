#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    CHECK_ARGUMENT(in->ndim() == 3, "RoPE: input must be 3D tensor [seqlen, nhead, d].");
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D tensor [seqlen].");
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: all tensors must be contiguous.");

    size_t seq_len = in->shape()[0];
    size_t num_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    CHECK_ARGUMENT(pos_ids->shape()[0] == seq_len, "RoPE: pos_ids length must match seqlen.");
    CHECK_ARGUMENT(head_dim % 2 == 0, "RoPE: head_dim must be even.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seq_len, num_heads, head_dim, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seq_len, num_heads, head_dim, theta);
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
