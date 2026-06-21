#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // in: (M, K), weight: (N, K), bias: (N,) or null, out: (M, N)
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];

    CHECK_ARGUMENT(in->ndim() == 2, "Linear: input must be 2D.");
    CHECK_ARGUMENT(weight->ndim() == 2, "Linear: weight must be 2D.");
    CHECK_ARGUMENT(weight->shape()[1] == K, "Linear: weight dimension mismatch with input.");
    CHECK_ARGUMENT(out->shape()[0] == M && out->shape()[1] == N, "Linear: output shape mismatch.");

    ASSERT(in->isContiguous() && weight->isContiguous() && out->isContiguous(),
           "Linear: all tensors must be contiguous.");

    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(),
                           bias ? bias->data() : nullptr, M, N, K, out->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(),
                           bias ? bias->data() : nullptr, M, N, K, out->dtype());
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
