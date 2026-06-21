#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void linear_(std::byte *out, const std::byte *in, const std::byte *weight,
             const std::byte *bias, size_t M, size_t N, size_t K) {
    const T *in_ptr = reinterpret_cast<const T *>(in);
    const T *w_ptr = reinterpret_cast<const T *>(weight);
    const T *b_ptr = reinterpret_cast<const T *>(bias);
    T *out_ptr = reinterpret_cast<T *>(out);

    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; k++) {
                    sum += llaisys::utils::cast<float>(in_ptr[m * K + k]) *
                           llaisys::utils::cast<float>(w_ptr[n * K + k]);//这样就转置了
                }
                if (b_ptr) {
                    sum += llaisys::utils::cast<float>(b_ptr[n]);
                }
                out_ptr[m * N + n] = llaisys::utils::cast<T>(sum);
            } else {
                T sum = T(0);
                for (size_t k = 0; k < K; k++) {
                    sum += in_ptr[m * K + k] * w_ptr[n * K + k];
                }
                if (b_ptr) {
                    sum += b_ptr[n];
                }
                out_ptr[m * N + n] = sum;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, size_t M, size_t N, size_t K,
            llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_<float>(out, in, weight, bias, M, N, K);
    case LLAISYS_DTYPE_BF16:
        return linear_<llaisys::bf16_t>(out, in, weight, bias, M, N, K);
    case LLAISYS_DTYPE_F16:
        return linear_<llaisys::fp16_t>(out, in, weight, bias, M, N, K);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
