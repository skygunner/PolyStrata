// src/poly/nemotronh.cpp - Nemotron-H / Nemotron 3.5 Bolt ("nemotron_h_moe"): Mamba-2 layers,
// NoPE GQA layers and latent ReLU^2 MoE layers with their own expert counts and FFN widths.
// Ported from justlovett0's PolyStrata-TeslaM40 cc72c9f5 (MIT).  The reference for the math is the
// pinned llama.cpp src/models/nemotron-h.cpp and mamba-base.cpp; the M40/SM52 work is not ported.
#include "nemotronh.hpp"

#include "nemotron_meta.hpp"
#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace strata::poly {
namespace {
constexpr size_t kScoreFloats = size_t(16) << 20;
}  // namespace

void NemotronH::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    n_mamba_ = n_attn_ = 0;
}

void NemotronH::load_geometry() {
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported Nemotron-H model: ") + what);
    };
    auto scalar = [&](const char* key, int64_t lo, int64_t hi) {
        const MetaValue* v = meta_in(*gg_, key);
        if (!v) throw std::runtime_error(std::string("nemotron_h_moe.") + key + " is missing");
        return nemotron_meta_int(*v, key, lo, hi);
    };
    auto arr = [&](const char* key, int lo, int hi) {
        const MetaValue* v = meta_in(*gg_, key);
        if (!v) return std::vector<int>((size_t) n_layer_, 0);
        return nemotron_meta_array(*v, key, n_layer_, lo, hi);
    };

    // Base dimensions first.  Every vector below is sized by n_layer_, and every later division and
    // allocation trusts these, so they are range-checked before either.
    n_layer_ = (int) scalar("block_count", 1, 512);
    n_embd_ = (int) scalar("embedding_length", 32, 1 << 20);
    need(n_embd_ % 32 == 0, "embedding_length is not a multiple of 32");
    n_head_ = (int) scalar("attention.head_count", 1, n_embd_);
    need(n_embd_ % n_head_ == 0, "embedding_length is not a multiple of attention.head_count");
    head_dim_ = n_embd_ / n_head_;
    rms_eps_ = shape_f("attention.layer_norm_rms_epsilon");
    need(std::isfinite(rms_eps_) && rms_eps_ > 0.f, "bad attention.layer_norm_rms_epsilon");

    const MetaValue* vocab = meta_in(*gg_, "vocab_size");
    const TensorInfo* te = gg_->find("token_embd.weight");
    if (vocab) n_vocab_ = (int) nemotron_meta_int(*vocab, "vocab_size", 1, 1 << 24);
    else n_vocab_ = te && te->shape.size() == 2 && te->shape[1] > 0 && te->shape[1] <= (1u << 24)
                        ? (int) te->shape[1] : 0;
    need(n_vocab_ > 0, "bad or missing vocab_size");

    const MetaValue* key_len = meta_in(*gg_, "attention.key_length");
    const MetaValue* val_len = meta_in(*gg_, "attention.value_length");
    need(!key_len || (int) nemotron_meta_int(*key_len, "attention.key_length", 1, 1 << 20) == head_dim_,
         "attention.key_length differs from embedding_length / head_count");
    need(!val_len || (int) nemotron_meta_int(*val_len, "attention.value_length", 1, 1 << 20) == head_dim_,
         "attention.value_length differs from embedding_length / head_count");

    kv_heads_ = arr("attention.head_count_kv", 0, n_head_);
    used_ = arr("expert_used_count", 0, kMaxUsed);
    expert_ff_ = arr("expert_feed_forward_length", 0, 1 << 20);
    n_kv_ = *std::max_element(kv_heads_.begin(), kv_heads_.end());
    n_used_ = *std::max_element(used_.begin(), used_.end());
    n_ff_exp_ = *std::max_element(expert_ff_.begin(), expert_ff_.end());
    n_expert_ = (int) scalar("expert_count", 1, 512);
    const MetaValue* shexp = meta_in(*gg_, "expert_shared_feed_forward_length");
    n_ff_shexp_ = shexp ? (int) nemotron_meta_int(*shexp, "expert_shared_feed_forward_length", 1, 1 << 20)
                        : n_ff_exp_;
    latent_width_ = (int) scalar("moe_latent_size", 32, n_embd_);
    conv_width_ = (int) scalar("ssm.conv_kernel", 1, 64);
    const int64_t inner = scalar("ssm.inner_size", 1, 1 << 20);
    const int64_t state_dim = scalar("ssm.state_size", 1, 4096);
    const int64_t groups = scalar("ssm.group_count", 1, 1 << 14);
    const int64_t ssm_heads = scalar("ssm.time_step_rank", 1, 1 << 14);
    const int64_t xbc = inner + 2 * groups * state_dim;
    need(xbc <= std::numeric_limits<int>::max() - inner - ssm_heads, "ssm dimensions overflow an int");
    inner_ = (int) inner;
    state_dim_ = (int) state_dim;
    groups_ = (int) groups;
    ssm_heads_ = (int) ssm_heads;
    xbc_ = (int) xbc;
    in_proj_ = (int) (xbc + inner + ssm_heads);
    const MetaValue* norm = meta_in(*gg_, "expert_weights_norm");
    const MetaValue* scale = meta_in(*gg_, "expert_weights_scale");
    const MetaValue* gating = meta_in(*gg_, "expert_gating_func");
    need(!gating || (gating->is_num() && gating->u == 2), "expert_gating_func is not sigmoid");
    need(!norm || norm->is_num(), "expert_weights_norm is not a number");
    need(!scale || scale->is_num(), "expert_weights_scale is not a number");
    w_norm_ = !norm || norm->u != 0;
    w_scale_ = scale ? (float) scale->num() : 1.f;
    need(std::isfinite(w_scale_) && w_scale_ > 0.f, "bad expert_weights_scale");

    need(n_embd_ % 32 == 0 && n_vocab_ > 0, "bad base geometry");
    need(n_kv_ > 0 && n_head_ % n_kv_ == 0, "bad attention heads");
    need(conv_width_ == 4 && state_dim_ == 128 && ssm_heads_ > 0 && groups_ > 0 && inner_ % ssm_heads_ == 0 &&
             ssm_heads_ % groups_ == 0,
         "unsupported Mamba-2 geometry");
    need(latent_width_ > 0 && latent_width_ % 32 == 0 && latent_width_ <= n_embd_ && n_ff_shexp_ > 0,
         "bad latent or shared-expert width");
    need(n_ff_exp_ > 0, "bad expert_feed_forward_length");
    need(n_used_ >= 1 && n_used_ <= kMaxUsed, "bad expert_used_count");
    for (int il = 0; il < n_layer_; ++il) {
        const int kv = kv_heads_[(size_t) il], used = used_[(size_t) il], ff = expert_ff_[(size_t) il];
        need(ff > 0 ? true : used == 0, "a layer has expert_used_count without expert_feed_forward_length");
        if (ff > 0) {
            need(used >= 1 && used <= n_used_, "bad per-layer expert count");
            need(ff <= n_ff_exp_, "expert_feed_forward_length above the file's maximum");
        } else {
            need(kv >= 0 && kv <= n_kv_, "bad per-layer head_count_kv");
        }
    }
    // One attention width for every attention layer: the k/v caches and the kernels take it.
    for (int il = 0; il < n_layer_; ++il)
        if (expert_ff_[(size_t) il] == 0 && kv_heads_[(size_t) il] > 0)
            need(kv_heads_[(size_t) il] == n_kv_, "attention layers with different head_count_kv");

    // Admission.  The layers' scratch and the expert tables hold one row of tokens, and the SSM state
    // leaves no snapshot for a guessed window.  Defaults are clamped; explicit incompatible values stop.
    if (opt_.prompt_batch != 1) {
        if (opt_.prompt_batch == 1024) opt_.prompt_batch = 1;   // the engine's default, this model's cap
        else throw std::runtime_error("--prompt-batch " + std::to_string(opt_.prompt_batch) +
                                      ": nemotron_h_moe runs one prompt token a call (its latent MoE scratch)");
    }
    if (opt_.window != 0 && opt_.window != 1)
        throw std::runtime_error("--window " + std::to_string(opt_.window) +
                                 ": nemotron_h_moe decodes one token at a time (no SSM window snapshots)");
    if (opt_.batch != 0)
        throw std::runtime_error("--batch " + std::to_string(opt_.batch) + ": nemotron_h_moe runs one sequence");

    draft_ = block_layer_ = false;
    block_ = nullptr;
    clamp_exp_.assign((size_t) n_layer_, INFINITY);
    clamp_shexp_ = clamp_exp_;
}

void NemotronH::load_nemotron_moe(Layer& L, Block& B, int il) {
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    const TensorInfo* up = tensor(il, "ffn_up_exps.weight");
    const TensorInfo* down = tensor(il, "ffn_down_exps.weight");
    if (!up || !down || up->shape.size() != 3 || down->shape.size() != 3)
        throw std::runtime_error(blk(il, "ffn_*_exps.weight") + " is missing or not rank 3");
    const int ne = (int) up->shape[2], ff = expert_ff_[(size_t) il], used = used_[(size_t) il];
    if (ne <= 0 || ne > n_expert_ || used < 1 || used > ne || (int) up->shape[0] != latent_width_ ||
        (int) up->shape[1] != ff || (int) down->shape[0] != ff || (int) down->shape[1] != latent_width_ ||
        (int) down->shape[2] != ne)
        throw std::runtime_error(blk(il, "ffn_*_exps.weight") + " has unexpected heterogeneous dimensions");
    L.moe = L.relu2 = true;
    L.expert_count = ne;
    L.expert_used = used;
    L.expert_ff = ff;
    L.expert_in = L.expert_out = latent_width_;
    L.gate_inp = mat("ffn_gate_inp.weight", n_embd_, ne);
    // One up matrix, not a gate/up pair: it is the slot's gate and its up, so gu2() stays the slot layout
    // the rest of the engine copies (gate | up | down).
    L.gu_type = up->type;
    L.gu_stride = L.gu_bytes = home(*up).payload(*up) / (size_t) ne;
    L.exp_gate = L.exp_up = home(*up).data(*up);
    L.u_type = L.gu_type;
    L.u_stride = L.u_bytes = L.gu_bytes;
    L.gu_step = L.gu2();
    L.d_type = down->type;
    L.d_bytes = home(*down).payload(*down) / (size_t) ne;
    L.exp_down = home(*down).data(*down);
    std::string err;
    if (!cpu_expert_types_ok((int) L.gu_type, (int) L.d_type, latent_width_, ff, err))
        throw std::runtime_error(blk(il, "ffn_*_exps.weight: ") + err);
    L.slot.assign((size_t) ne, nullptr);
    B.latent_down = mat("ffn_latent_down.weight", n_embd_, latent_width_);
    B.latent_up = mat("ffn_latent_up.weight", latent_width_, n_embd_);
    B.shared_up = mat("ffn_up_shexp.weight", n_embd_, n_ff_shexp_);
    B.shared_down = mat("ffn_down_shexp.weight", n_ff_shexp_, n_embd_);
    const TensorInfo* bias = tensor(il, "exp_probs_b.bias");
    if (!bias || bias->type != kTypeF32 || (int) bias->elements() != ne)
        throw std::runtime_error(blk(il, "exp_probs_b.bias") + " is missing or has the wrong shape");
    B.bias.resize((size_t) ne);
    std::memcpy(B.bias.data(), home(*bias).data(*bias), (size_t) ne * sizeof(float));
    ++n_moe_;
}

void NemotronH::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    // Nemotron packs some ssm projections in NVFP4 with a per-tensor F32 .scale; llama.cpp multiplies the
    // projection's result by it (build_lora_mm(..., ssm_in_s)).  The weight bytes are untouched; the factor
    // is applied on the result, as llama.cpp does.  A file without the tensor keeps the plain product.
    auto scale_of = [&](const char* sname) {
        const TensorInfo* s = tensor(il, sname);
        if (!s) return Dev{};
        if (s->type != kTypeF32 || s->elements() != 1)
            throw std::runtime_error(s->name + " is not one F32 value");
        return upload(s, blk(il, sname), 1, 1, kF32);
    };
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.fixed = true;
    if (expert_ff_[(size_t) il] > 0) {
        B.kind = Moe;
        load_nemotron_moe(L, B, il);
    } else if (kv_heads_[(size_t) il] > 0) {
        B.kind = Attention;
        B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * head_dim_);
        B.wk = mat("attn_k.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.wv = mat("attn_v.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.wo = mat("attn_output.weight", (int64_t) n_head_ * head_dim_, n_embd_);
        ++n_attn_;
    } else {
        B.kind = Mamba;
        B.ssm_in = mat("ssm_in.weight", n_embd_, in_proj_);
        B.ssm_in_s = scale_of("ssm_in.scale");
        B.conv = f32("ssm_conv1d.weight", conv_width_, xbc_);
        B.conv_bias = f32("ssm_conv1d.bias", xbc_);
        B.dt_bias = f32("ssm_dt.bias", ssm_heads_);
        B.a = f32("ssm_a", 1, ssm_heads_);
        B.d = f32("ssm_d", 1, ssm_heads_);
        B.ssm_norm = f32("ssm_norm.weight", inner_ / groups_, groups_);
        B.wo = mat("ssm_out.weight", inner_, n_embd_);
        B.ssm_out_s = scale_of("ssm_out.scale");
        ++n_mamba_;
    }
}

size_t NemotronH::recurrent_floats() const {
    return (size_t) n_mamba_ * ((size_t) 3 * xbc_ + (size_t) inner_ * state_dim_);
}

void NemotronH::alloc_caches() {
    float* state = state_;
    const size_t kv_row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.kind == Mamba) {
            B.conv_state = state; state += (size_t) 3 * xbc_;
            B.ssm_state = state; state += (size_t) inner_ * state_dim_;
        } else if (B.kind == Attention) {
            B.keys = (uint16_t*) kv_alloc(kv_row, 1);
            B.vals = (uint16_t*) kv_alloc(kv_row, 1);
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

void NemotronH::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    proj_ = fbuf(n * in_proj_);
    conv_out_ = fbuf(n * xbc_);
    mamba_out_ = fbuf(n * inner_);
    q_ = fbuf(heads); k_ = fbuf(kv); v_ = fbuf(kv); att_ = fbuf(heads);
    score_n_ = std::max((size_t) chunk_ * n_head_ * opt_.max_context,
                        attend_split_floats(chunk_, n_head_, opt_.max_context, head_dim_));
    score_ = fbuf(score_n_);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);
    latent_ = fbuf(n * latent_width_);
    shared_ = fbuf(n * n_embd_);
}

void NemotronH::save_layer(int il, int64_t, uint8_t* dst) {
    const Block& B = blocks_[(size_t) il];
    if (!B.conv_state) return;
    // conv_state and ssm_state are contiguous in state_
    const size_t n = (size_t) 3 * xbc_ + (size_t) inner_ * state_dim_;
    ck(cudaMemcpy(dst + (size_t) (B.conv_state - state_) * sizeof(float), B.conv_state, n * sizeof(float),
                  cudaMemcpyDeviceToHost), "save Nemotron Mamba state");
}

size_t NemotronH::card_row_bytes() const { return (size_t) n_attn_ * 2 * n_kv_ * head_dim_ * sizeof(uint16_t); }

void NemotronH::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context) throw std::logic_error("save rows outside context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    for (const Block& B : blocks_) if (B.keys) for (const uint16_t* p : {B.keys, B.vals}) {
        ck(cudaMemcpy(dst, (const uint8_t*) p + at, bytes, cudaMemcpyDeviceToHost), "save rows"); dst += bytes;
    }
}

void NemotronH::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load rows outside saved context");
    cover(first + n); ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row;
    const size_t bytes = (size_t) n * row, skip = (size_t) (first - from) * row, all = (size_t) count * row;
    for (const Block& B : blocks_) if (B.keys) for (uint16_t* p : {B.keys, B.vals}) {
        ck(cudaMemcpy((uint8_t*) p + at, src + skip, bytes, cudaMemcpyHostToDevice), "load rows"); src += all;
    }
}

void NemotronH::mamba(Block& B, int n) {
    // The ssm_in projection interleaves z / xBC / dt per token with in_proj_ between rows, while the
    // convolution and the scan read tight rows.  Admission holds prompt_batch at 1, so every call here is
    // one token; a batch of more would need strided views (not ported).
    if (n != 1) throw std::logic_error("nemotron_h_moe Mamba layers run one token a call");
    lin(B.ssm_in, xn_, proj_, n, xq_);
    if (B.ssm_in_s.p) scale_rows(proj_, in_proj_, in_proj_, n, nullptr, fp(B.ssm_in_s), false, st_);
    float* z = proj_;
    float* xbc = proj_ + inner_;
    float* dt = proj_ + inner_ + xbc_;
    conv4_silu(xbc, B.conv_state, fp(B.conv), conv_out_, xbc_, n, st_,
               snap_bits_ ? snap_at_ + (B.conv_state - state_) : nullptr, rec_n_, snap_bits_, fp(B.conv_bias));
    add_bias(fp(B.dt_bias), dt, ssm_heads_, n, st_);
    const float* b = conv_out_ + inner_;
    const float* c = b + groups_ * state_dim_;
    mamba2_step(conv_out_, dt, fp(B.a), b, c, fp(B.d), z, fp(B.ssm_norm), B.ssm_state, mamba_out_,
                inner_, ssm_heads_, groups_, state_dim_, n, rms_eps_, st_);
    lin(B.wo, mamba_out_, ao_, n);
    if (B.ssm_out_s.p) scale_rows(ao_, n_embd_, n_embd_, n, nullptr, fp(B.ssm_out_s), false, st_);
}

void NemotronH::attention(Block& B, int64_t pos, int n) {
    const long long* at = recording_ ? at_ : nullptr;
    lin(B.wq, xn_, q_, n, xq_); lin(B.wk, xn_, k_, n, xq_); lin(B.wv, xn_, v_, n, xq_);
    if (at) { store_f16_at(k_, B.keys, n_kv_ * head_dim_, at, n, 0, st_); store_f16_at(v_, B.vals, n_kv_ * head_dim_, at, n, 0, st_); }
    else { store_f16(k_, B.keys, n_kv_ * head_dim_, pos, n, st_); store_f16(v_, B.vals, n_kv_ * head_dim_, pos, n, st_); }
    const float scale = 1.f / std::sqrt((float) head_dim_);
    if (n >= kStageRows)
        attend_batch(blas_, q_, B.keys, B.vals, pos, n, score_, (size_t) chunk_ * n_head_ * opt_.max_context,
                     awork_, att_, head_dim_, n_head_, n_kv_, scale, st_);
    else
        attend(q_, B.keys, B.vals, pos, n, score_, chunk_, att_, head_dim_, n_head_, n_kv_, scale, st_, 0, 0,
               false, at, at ? room(n) : 0, score_n_);
    lin(B.wo, att_, ao_, n);
}

int NemotronH::head_key(int il, int n) const { return blocks_[(size_t) il].kind == Attention ? room_key(n) : 0; }

void NemotronH::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    rms_norm(r_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    if (B.kind == Mamba) mamba(B, n);
    else if (B.kind == Attention) attention(B, pos_, n);
    else {
        lin(L.gate_inp, xn_, rl_, n, xq_);
        lin(B.shared_up, xn_, g_, n, xq_);
        relu2_rows(g_, h_, n_ff_shexp_, n, st_);
        lin(B.shared_down, h_, shared_, n);
        lin(B.latent_down, xn_, latent_, n, xq_);
        ck(cudaMemcpyAsync(xn_, latent_, (size_t) n * latent_width_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
           "Nemotron latent input");
        kernels::native_quantize_q8_1(xn_, xq_, latent_width_ * n, 1, st_);
        copy2(rl_, h_rl_, n * L.expert_count, xn_, h_x_, n * latent_width_, st_);
        return;
    }
    axpy(ao_, 1.f, r_, n * n_embd_, true, st_);
}

void NemotronH::moe_shared(const Layer&, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * latent_width_ * sizeof(float), cs(st_)), "Nemotron experts");
}

void NemotronH::layer_tail(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    if (!L.moe) return;
    lin(B.latent_up, fo_, ao_, n);
    axpy(shared_, 1.f, ao_, n * n_embd_, true, st_);
    axpy(ao_, 1.f, r_, n * n_embd_, true, st_);
}

void NemotronH::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

void NemotronH::route(int il, const float* logits, int32_t, int* ids, float* w, float* tmp) const {
    const Layer& L = layers_[(size_t) il];
    nemotron_route(logits, L.expert_count, L.expert_used, blocks_[(size_t) il].bias.data(), w_norm_, w_scale_, ids,
                   w, tmp);
}

}  // namespace strata::poly
