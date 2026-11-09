// NVIDIA Nemotron-H / Nemotron 3: hybrid Mamba-2, GQA and heterogeneous latent MoE (GGUF
// "nemotron_h_moe").  Text only.  Ported from justlovett0's PolyStrata-TeslaM40 commit cc72c9f5
// (MIT, the same terms as this file's repository); the Tesla M40 / SM52 work is not part of this
// port.  Runs one prompt token a call and one decode token at a time; see load_geometry().
#pragma once

#include "engine.hpp"
#include "nemotron_route.hpp"

namespace strata::poly {

class NemotronH final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override;
    // SSM state is not written into the window's snapshots (mamba2_step leaves only the final
    // state), so a guessed window could not be rolled back.  Admission keeps window == 1.
    bool snaps_spread() const override { return false; }
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override { return in_proj_; }
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    void moe_shared(const Layer& L, int il, int n) override;
    int head_key(int il, int n) const override;

private:
    enum Kind : uint8_t { Mamba, Attention, Moe };
    struct Block {
        Kind kind = Mamba;
        Dev wo;
        Dev ssm_in, conv, conv_bias, dt_bias, a, d, ssm_norm;
        Dev ssm_in_s, ssm_out_s;   ///< Nemotron's F32 projection scales; .p is null when the file has none
        float* conv_state = nullptr;
        float* ssm_state = nullptr;
        Dev wq, wk, wv;
        uint16_t* keys = nullptr;
        uint16_t* vals = nullptr;
        Dev latent_down, latent_up, shared_up, shared_down;
        std::vector<float> bias;
    };

    void mamba(Block& B, int n);
    void attention(Block& B, int64_t pos, int n);
    void load_nemotron_moe(Layer& L, Block& B, int il);
    std::vector<Block> blocks_;
    std::vector<int> kv_heads_, used_, expert_ff_;
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0;
    int conv_width_ = 4, inner_ = 0, state_dim_ = 0, groups_ = 0, ssm_heads_ = 0, xbc_ = 0, in_proj_ = 0;
    int latent_width_ = 0, n_mamba_ = 0, n_attn_ = 0;
    bool w_norm_ = true;
    float w_scale_ = 1.f;
    int chunk_ = 1;
    size_t score_n_ = 0;

    float *proj_ = nullptr, *conv_out_ = nullptr, *mamba_out_ = nullptr;
    float *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr, *score_ = nullptr, *awork_ = nullptr;
    float *latent_ = nullptr, *shared_ = nullptr;
};

}  // namespace strata::poly
