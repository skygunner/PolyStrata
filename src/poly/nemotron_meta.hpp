// src/poly/nemotron_meta.hpp - the checks Nemotron-H's loader puts on its metadata before it uses a
// number: integer type, range, and arrays of them.  Header only so a host test can feed malformed
// values through the same checks without a model or a card.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::poly {

/// One integer of the model's metadata: an integer type (not F32/F64/BOOL/STRING/ARRAY), in [lo, hi].
inline int64_t nemotron_meta_int(const MetaValue& v, const char* key, int64_t lo, int64_t hi) {
    switch (v.type) {
    case MetaType::U8:
    case MetaType::U16:
    case MetaType::U32:
    case MetaType::U64:
    case MetaType::I8:
    case MetaType::I16:
    case MetaType::I32:
    case MetaType::I64:
        break;
    default:
        throw std::runtime_error(std::string("nemotron_h_moe.") + key + " is not an integer");
    }
    const int64_t x = (int64_t) v.u;
    if (x < lo || x > hi)
        throw std::runtime_error(std::string("nemotron_h_moe.") + key + " = " + std::to_string(x) + ", expected " +
                                 std::to_string(lo) + ".." + std::to_string(hi));
    return x;
}

/// One integer a layer, `n_layer` of them: an ARRAY with at least that many entries, each in [lo, hi].
inline std::vector<int> nemotron_meta_array(const MetaValue& v, const char* key, int n_layer, int lo, int hi) {
    if (v.type != MetaType::ARRAY || v.items.size() < (size_t) n_layer)
        throw std::runtime_error(std::string("nemotron_h_moe.") + key + " is not an array of " +
                                 std::to_string(n_layer) + " integers");
    std::vector<int> out((size_t) n_layer);
    for (int i = 0; i < n_layer; ++i) out[(size_t) i] = (int) nemotron_meta_int(v.items[(size_t) i], key, lo, hi);
    return out;
}

}  // namespace strata::poly
