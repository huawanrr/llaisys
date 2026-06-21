#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];

    CHECK_ARGUMENT(in->ndim() == 2, "RmsNorm: input must be 2D.");
    CHECK_ARGUMENT(weight->ndim() == 1, "RmsNorm: weight must be 1D.");
    CHECK_ARGUMENT(weight->shape()[0] == K, "RmsNorm: weight dimension mismatch with input.");
    CHECK_ARGUMENT(out->shape()[0] == M && out->shape()[1] == K, "RmsNorm: output shape mismatch.");

    ASSERT(in->isContiguous() && weight->isContiguous() && out->isContiguous(),
           "RmsNorm: all tensors must be contiguous.");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), M, K, eps, out->dtype());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), M, K, eps, out->dtype());
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
