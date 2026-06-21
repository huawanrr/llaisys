#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>
#include <limits>

template <typename T>
void self_attention_(std::byte *attn_val_raw, const std::byte *q_raw, const std::byte *k_raw, const std::byte *v_raw,
                     size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                     size_t d, size_t dv, float scale) {

    const T *q = reinterpret_cast<const T *>(q_raw);
    const T *k = reinterpret_cast<const T *>(k_raw);
    const T *v = reinterpret_cast<const T *>(v_raw);
    T *out = reinterpret_cast<T *>(attn_val_raw);

    size_t heads_per_kv = nhead / nkvhead;

    for (size_t h = 0; h < nhead; h++) {
        size_t kv_h = h / heads_per_kv; 

        for (size_t i = 0; i < seqlen; i++) {
            
            size_t causal_end = total_len - seqlen + i + 1; 

            std::vector<float> attn_scores(total_len);

            for (size_t j = 0; j < total_len; j++) {
                if (j < causal_end) {
                    float dot = 0.0f;
                    for (size_t dd = 0; dd < d; dd++) {
                        float q_val, k_val;
                        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                            q_val = llaisys::utils::cast<float>(q[i * nhead * d + h * d + dd]);
                            k_val = llaisys::utils::cast<float>(k[j * nkvhead * d + kv_h * d + dd]);
                        } else {
                            q_val = static_cast<float>(q[i * nhead * d + h * d + dd]);
                            k_val = static_cast<float>(k[j * nkvhead * d + kv_h * d + dd]);
                        }
                        dot += q_val * k_val;
                    }
                    attn_scores[j] = dot * scale;
                } else {
                    attn_scores[j] = -std::numeric_limits<float>::infinity();
                }
            }
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < total_len; j++) {
                if (attn_scores[j] > max_score) {
                    max_score = attn_scores[j];
                }
            }

            float sum_exp = 0.0f;
            for (size_t j = 0; j < total_len; j++) {
                attn_scores[j] = std::exp(attn_scores[j] - max_score);
                sum_exp += attn_scores[j];
            }

            for (size_t j = 0; j < total_len; j++) {
                attn_scores[j] /= sum_exp;
            }

            for (size_t dd = 0; dd < dv; dd++) {
                float acc = 0.0f;
                for (size_t j = 0; j < total_len; j++) {
                    float v_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v[j * nkvhead * dv + kv_h * dv + dd]);
                    } else {
                        v_val = static_cast<float>(v[j * nkvhead * dv + kv_h * dv + dd]);
                    }
                    acc += attn_scores[j] * v_val;
                }
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[i * nhead * dv + h * dv + dd] = llaisys::utils::cast<T>(acc);
                } else {
                    out[i * nhead * dv + h * dv + dd] = static_cast<T>(acc);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                    size_t d, size_t dv, float scale, llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_<float>(attn_val, q, k, v, seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_<llaisys::bf16_t>(attn_val, q, k, v, seqlen, total_len, nhead, nkvhead, d, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_<llaisys::fp16_t>(attn_val, q, k, v, seqlen, total_len, nhead, nkvhead, d, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
