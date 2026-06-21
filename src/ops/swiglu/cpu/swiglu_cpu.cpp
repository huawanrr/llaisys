#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void swiglu_(std::byte *out_raw, const std::byte *gate_raw, const std::byte *up_raw, size_t numel) {
    T *out = reinterpret_cast<T *>(out_raw);
    const T *gate = reinterpret_cast<const T *>(gate_raw);
    const T *up = reinterpret_cast<const T *>(up_raw);

    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float g = llaisys::utils::cast<float>(gate[i]);
            float u = llaisys::utils::cast<float>(up[i]);
            float sigmoid_g = 1.0f / (1.0f + std::exp(-g));
            float result = u * g * sigmoid_g;
            out[i] = llaisys::utils::cast<T>(result);
        } else {
            T g = gate[i];
            T u = up[i];
            T sigmoid_g = T(1) / (T(1) + std::exp(-g));
            out[i] = u * g * sigmoid_g;
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_<llaisys::bf16_t>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_<llaisys::fp16_t>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
