#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>  
#include <cstdint>  

namespace llaisys::ops::cpu {
void embedding(std::byte *output, const std::byte *index, const std::byte *weight,
               size_t num_embeddings, size_t embedding_dim, size_t num_indices,
               llaisysDataType_t type) {
  const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);
  size_t row_bytes = embedding_dim * llaisys::utils::dsize(type);

  for (size_t i = 0; i < num_indices; i++) {
    int64_t idx = idx_ptr[i];
    ASSERT(idx >= 0 && static_cast<size_t>(idx) < num_embeddings,
           "embedding index out of range");
    std::memcpy(output + i * row_bytes,
                weight + idx * row_bytes,
                row_bytes);
  }
}
} 