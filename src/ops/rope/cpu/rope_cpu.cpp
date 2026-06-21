#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>
#include <type_traits>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t seq_len, size_t num_heads, size_t head_dim, float theta) {
    const size_t half_dim = head_dim / 2;

    for (size_t s = 0; s < seq_len; s++) {
        const float m = static_cast<float>(pos_ids[s]);

        for (size_t h = 0; h < num_heads; h++) {
            const size_t base = s * num_heads * head_dim + h * head_dim;

            for (size_t j = 0; j < half_dim; j++) {
                float freq = m / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);

                // a = x[j], b = x[d/2 + j]  
                float x0, x1;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    x0 = llaisys::utils::cast<float>(in[base + j]);
                    x1 = llaisys::utils::cast<float>(in[base + half_dim + j]);
                } else {
                    x0 = static_cast<float>(in[base + j]);
                    x1 = static_cast<float>(in[base + half_dim + j]);
                }

                float out0 = x0 * cos_val - x1 * sin_val;
                float out1 = x1 * cos_val + x0 * sin_val;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[base + j] = llaisys::utils::cast<T>(out0);
                    out[base + half_dim + j] = llaisys::utils::cast<T>(out1);
                } else {
                    out[base + j] = static_cast<T>(out0);
                    out[base + half_dim + j] = static_cast<T>(out1);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, 
          llaisysDataType_t dtype, 
          size_t seq_len, size_t num_heads, size_t head_dim, float theta) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seq_len, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seq_len, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seq_len, num_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
