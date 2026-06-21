#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *output, const std::byte *index, const std::byte *weight,
               size_t num_embeddings, size_t embedding_dim, size_t num_indices,
               llaisysDataType_t type);
}