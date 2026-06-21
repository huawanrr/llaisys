#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(std::byte *out, const std::byte *in, const std::byte *weight,
               size_t M, size_t K, float eps) {
    const T *in_ptr = reinterpret_cast<const T *>(in);
    const T *w_ptr = reinterpret_cast<const T *>(weight);
    T *out_ptr = reinterpret_cast<T *>(out);

    for (size_t m = 0; m < M; m++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float mean_sq = 0.0f;
            for (size_t k = 0; k < K; k++) {
                float val = llaisys::utils::cast<float>(in_ptr[m * K + k]);
                mean_sq += val * val;
            }
            mean_sq /= static_cast<float>(K);

            float rms = 1.0f / std::sqrt(mean_sq + eps);

            for (size_t k = 0; k < K; k++) {
                float val = llaisys::utils::cast<float>(in_ptr[m * K + k]);
                float w = llaisys::utils::cast<float>(w_ptr[k]);
                out_ptr[m * K + k] = llaisys::utils::cast<T>(val * rms * w);
            }
        } else {
            T mean_sq = T(0);
            for (size_t k = 0; k < K; k++) {
                T val = in_ptr[m * K + k];
                mean_sq += val * val;
            }//算平方
            mean_sq /= static_cast<T>(K);

            T rms = T(1) / std::sqrt(mean_sq + static_cast<T>(eps));

            for (size_t k = 0; k < K; k++) {
                out_ptr[m * K + k] = in_ptr[m * K + k] * rms * w_ptr[k];//每一行内积
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              size_t M, size_t K, float eps, llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_<float>(out, in, weight, M, K, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_<llaisys::bf16_t>(out, in, weight, M, K, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_<llaisys::fp16_t>(out, in, weight, M, K, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
