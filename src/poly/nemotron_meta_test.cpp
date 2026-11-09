// src/poly/nemotron_meta_test.cpp - the metadata checks before Nemotron-H uses a number: a bad type,
// a bad range or a short per-layer array must fail with a message, not size a vector or divide by it.
#include "nemotron_meta.hpp"

#include <cstdio>
#include <string>

using namespace strata;
using strata::poly::nemotron_meta_array;
using strata::poly::nemotron_meta_int;

static int failures = 0;

template <class F> static void expect_throw(F&& f, const char* what, const char* key = nullptr) {
    try {
        f();
        std::fprintf(stderr, "FAIL: %s did not throw\n", what);
        ++failures;
    } catch (const std::exception& e) {
        if (key && std::string(e.what()).find(key) == std::string::npos) {
            std::fprintf(stderr, "FAIL: %s threw without naming %s: %s\n", what, key, e.what());
            ++failures;
        }
    }
}

static MetaValue num(MetaType t, uint64_t u) {
    MetaValue v;
    v.type = t;
    v.u = u;
    return v;
}

template <class T> static void put(std::vector<uint8_t>& b, T v) {
    uint8_t bytes[sizeof(T)];
    std::memcpy(bytes, &v, sizeof(T));
    b.insert(b.end(), bytes, bytes + sizeof(T));
}

// A GGUF array header and its values, as read_value() parses them from a file's metadata.
static MetaValue parse_array(MetaType elem, uint64_t count) {
    std::vector<uint8_t> b;
    put<uint32_t>(b, (uint32_t) elem);
    put<uint64_t>(b, count);
    for (uint64_t i = 0; i < count; ++i) {
        if (elem == MetaType::U32) put<uint32_t>(b, (uint32_t) (i % 3));
        else put<float>(b, (float) i);
    }
    Cursor c(b.data(), b.size());
    return read_value(c, MetaType::ARRAY);
}

static MetaValue parse_i32(int32_t v) {
    std::vector<uint8_t> b;
    put<int32_t>(b, v);
    Cursor c(b.data(), b.size());
    return read_value(c, MetaType::I32);
}

int main() {
    // A valid scalar and array go through.
    if (nemotron_meta_int(num(MetaType::U32, 88), "block_count", 1, 512) != 88) {
        std::fprintf(stderr, "FAIL: a valid block_count did not load\n");
        ++failures;
    }
    MetaValue arr;
    arr.type = MetaType::ARRAY;
    arr.elem = MetaType::U32;
    arr.count = 3;
    arr.items = {num(MetaType::U32, 0), num(MetaType::U32, 9), num(MetaType::U32, 2)};
    const auto per_layer = nemotron_meta_array(arr, "expert_used_count", 3, 0, 32);
    if (per_layer.size() != 3 || per_layer[1] != 9 || per_layer[2] != 2) {
        std::fprintf(stderr, "FAIL: a valid per-layer array did not load\n");
        ++failures;
    }

    // The malformed values the old code cast or dereferenced.
    expect_throw([&] { nemotron_meta_int(num(MetaType::ARRAY, 0), "block_count", 1, 512); }, "array as scalar",
                 "block_count");
    expect_throw([&] { nemotron_meta_int(num(MetaType::F32, 0), "block_count", 1, 512); }, "float as scalar",
                 "block_count");
    expect_throw([&] { nemotron_meta_int(num(MetaType::U32, 0), "block_count", 1, 512); }, "zero layers",
                 "block_count");
    expect_throw([&] { nemotron_meta_int(num(MetaType::I32, (uint64_t) -1), "block_count", 1, 512); },
                 "negative layers", "block_count");
    expect_throw([&] { nemotron_meta_int(num(MetaType::U32, 513), "expert_count", 1, 512); }, "range above",
                 "expert_count");

    expect_throw([&] { nemotron_meta_array(arr, "expert_used_count", 4, 0, 32); }, "short array",
                 "expert_used_count");
    MetaValue floats = arr;
    floats.items[1] = num(MetaType::F32, 0);
    expect_throw([&] { nemotron_meta_array(floats, "expert_used_count", 3, 0, 32); }, "float in array",
                 "expert_used_count");
    MetaValue high = arr;
    high.items[2] = num(MetaType::U32, 33);
    expect_throw([&] { nemotron_meta_array(high, "expert_used_count", 3, 0, 32); }, "above range in array",
                 "expert_used_count");
    expect_throw([&] { nemotron_meta_array(num(MetaType::U32, 3), "expert_used_count", 3, 0, 32); },
                 "scalar as array", "expert_used_count");

    // Parsed from GGUF metadata bytes, not hand-built values: a U32 array passes, an F32 array and a
    // negative I32 scalar (the crash the old cast allowed) do not.
    const auto parsed = parse_array(MetaType::U32, 3);
    const auto parsed_ok = nemotron_meta_array(parsed, "expert_used_count", 3, 0, 32);
    if (parsed_ok.size() != 3 || parsed_ok[0] != 0 || parsed_ok[2] != 2) {
        std::fprintf(stderr, "FAIL: a parsed U32 array did not load\n");
        ++failures;
    }
    expect_throw([&] { nemotron_meta_array(parse_array(MetaType::F32, 3), "expert_used_count", 3, 0, 32); },
                 "parsed F32 array", "expert_used_count");
    expect_throw([&] { nemotron_meta_int(parse_i32(-1), "block_count", 1, 512); }, "parsed negative block_count",
                 "block_count");

    if (failures == 0) std::puts("nemotron metadata: valid values load, malformed values name the key");
    return failures == 0 ? 0 : 1;
}
