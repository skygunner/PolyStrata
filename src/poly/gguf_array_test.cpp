// src/poly/gguf_array_test.cpp - the reader keeps whole per-layer geometry arrays (> 64 items)
// and still samples very long ones (tokenizer lists).
#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace strata;

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

template <class T> static void put(std::vector<uint8_t>& b, T v) {
    uint8_t bytes[sizeof(T)];
    std::memcpy(bytes, &v, sizeof(T));
    b.insert(b.end(), bytes, bytes + sizeof(T));
}

// A U32 array header followed by `n` values, as read_value() takes it.
static MetaValue read_u32_array(uint64_t n) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, (uint32_t) MetaType::U32);
    put<uint64_t>(b, n);
    for (uint64_t i = 0; i < n; ++i) put<uint32_t>(b, (uint32_t) (i * 3));
    Cursor c(b.data(), b.size());
    return read_value(c, MetaType::ARRAY);
}

int main() {
    {
        const MetaValue v = read_u32_array(88);   // Nemotron-H's 88 layers
        check(v.count == 88 && v.items.size() == 88, "88 entries are kept whole");
        check(v.items[87].u == 87 * 3, "the last entry is the file's last");
    }
    {
        const MetaValue v = read_u32_array(64);   // every current model
        check(v.count == 64 && v.items.size() == 64, "64 entries stay complete");
    }
    {
        const MetaValue v = read_u32_array(2000);   // a tokenizer list: still a sample
        check(v.count == 2000 && v.items.size() == 1024, "2000 entries are sampled at 1024");
        check(v.items[1023].u == 1023 * 3, "the sample is the first 1024");
    }
    if (failures == 0) std::puts("gguf arrays: 88 kept, 64 kept, 2000 sampled at 1024");
    return failures == 0 ? 0 : 1;
}
