# Nemotron 3.5 Bolt (`nemotron_h_moe`)

PolyStrata's `strata-poly` engine admits the GGUF architecture `nemotron_h_moe` (NVIDIA Nemotron 3.5
Bolt, text only): a hybrid of Mamba-2 layers, NoPE grouped-query attention layers and latent
Mixture-of-Experts layers with ReLU² experts. It reads the GGUF in place; no repack. The backend
list's registration is not a claim about other Nemotron releases: another architecture name is
refused, and only the official pack below has been fed to this port.

## What runs

- Architecture `nemotron_h_moe`, text only. No MTP draft block, no multimodal input, no
  speculative decoding.
- Per-layer expert geometry from the file: expert counts, used counts, FFN widths, and a latent
  width narrower than the embedding. Two layers of one file may disagree.
- The tensor types go through the shared dequantizer and CPU-expert paths; the file this code was
  written against is the official IQ2_XXS + NVFP4 pack below, and no other quantization of this
  architecture has been exercised. NVFP4 `ssm_in` / `ssm_out` are computed on the card from the file's
  own 36-byte blocks (`native_mmvq` type 40, the expression of `vec_dot_nvfp4_q8_1`), with the BF16
  expansion kept as the fallback, and their F32 `.scale` is applied to the projection result, as
  llama.cpp's `build_lora_mm(..., ssm_in_s)` does. The source supplied the architecture implementation;
  support for the official checkpoint's NVFP4 projection scales was added during this port.

## Running it

```sh
strata-poly --model /path/to/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf --serve --max-context 4096
```

The official file is 22,018,326,528 bytes (22.02 GB / 20.51 GiB). The "24GB" in its name is
NVIDIA's label, not its measured size.

The initial architecture cap is **one prompt token a call, one decode token, one sequence**.
`--prompt-batch 1` is supported; the default value 1024 is also accepted and lowered to 1,
including when explicitly supplied. Other prompt batch sizes, `--window` above 1 and `--batch`
above 0 are refused. The default window becomes 1 automatically. Mamba state has no per-token
rollback snapshots, so speculative windows are not supported.

Serial continuation, reset and in-memory conversation checkpoints are supported. Disk-session
`SAVE` and `RESTORE` commands are not implemented by `strata-poly`, for any architecture.
Use the native engine or an existing server run config; this port adds no `setup.sh` family.

## Model files this was written against

- Official checkpoint: `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` —
  22,018,326,528 bytes (22.02 GB / 20.51 GiB), SHA-256
  `91d63363e782c8a3d5f4be42816eba8e0d42dd2530bd07a7d0d2dc440599af75`, Hugging Face revision
  `e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246`. "24GB" is NVIDIA's label, not its measured size.
  (`general.architecture = nemotron_h_moe`, 88 layers, 4096 embedding, 131072 vocabulary,
  `expert_count` 496, per-layer used counts 9, per-layer FFN 1536..2560, latent 1024,
  `ssm.inner_size` 8192, `ssm.state_size` 128, `ssm.group_count` 8, `ssm.time_step_rank` 128,
  `ssm.conv_kernel` 4, shared expert FFN 5376, `expert_weights_scale` 5.0). The tensor and scalar
  inventory is in `official-inventory.json` (843 tensors: 114 Q8_0, 80 NVFP4, 489 F32, 160
  IQ2_XXS). The weight file itself is unchanged by this port; only the engine reading it was.
- Source of the port: PolyStrata-TeslaM40 commit `cc72c9f5` by justlovett0 (MIT). Only the
  text-only architecture work is ported; the Tesla M40 / SM52 compatibility, Windows launchers
  and Maxwell presets are not. **No Q4 file of this architecture has been run by this port.**

## Validation on 2026-11-10

Measured on an Intel Core Ultra 7 265K (20 cores, 13 expert threads), 46 GiB of usable RAM
(48 GB installed), an NVIDIA GeForce RTX 3080 Ti 12 GiB, Linux, CUDA 12.8, g++ 14.3.1, SM 86. The model file
lives on a rotational HDD (`/dev/sda1`), which is why the first cold request is slow. The pinned
ggml source is `3cf03257f219afbe7334045ff7c6a06ac68c627d`; the reference llama.cpp binary is a
separate clean build at source `5628c7338963b00a598f2a588fd09a79af6d70db`.

The implementation is commit `1a3572b6130abdede73fdf7345c3af1717a85c87`. Raw logs and JSON files
named here are retained on the validation host at
`/mnt/storage/PolyStrata-bolt-validation/20261110/`. Full-model checks used a 4096-token context,
13 expert threads and a 512 MiB VRAM reserve. Token/probability comparisons froze the expert
cache; the dynamic low-RAM and HTTP checks did not.

- Host: router arithmetic, malformed per-layer metadata, GGUF arrays above 64 entries, the CPU
  ReLU² expert path (including a down projection narrower than the token width and a negative
  gate), and the existing CPU expert pool pass (`poly_nemotron_route_test`,
  `poly_nemotron_meta_test`, `poly_gguf_array_test`, `poly_cpu_experts_relu2_test`,
  `poly_workers_test`). The same-type gate/down FF alignment of `cpu_expert_types_ok` is covered
  there too. A fresh CPU-only build of the whole tree ran 48 tests: 47 passed, 1 expected skip
  (`expert_multi_test`), 0 failed, 7.04 s.
- GPU, run on the card: `poly_relu2_test` 14/14 exact; `poly_conv4_bias_test` max abs
  8.94e-8 (biased and unbiased convolution against the host, null bias equals the plain call);
  `poly_mamba2_test` with the actual inner 8192 / heads 128 / groups 8 case, max abs 3.34e-6 and
  0 for the chunked prompt against one call; `poly_nemotron_scale_test` exact on the projection
  result and on the plain product without a scale; the existing router test, 6 cases, 0 pick
  differences. All five under compute-sanitizer: 0 errors.
- Qwen3.6 regression unchanged by this port: 10 prompts, 551 generated tokens, exact match to the
  same-day pristine baseline (`base-qwen-greedy.json`).
- Python registry and config: 66 focused tests pass.
- Actual HTTP on loopback 18086: `READY`, then `cobalt`, with 27 and 58 prompt tokens;
  the second request reused 29 cached tokens (`final-http-chat.json`). Separately, the native
  protocol restored 57 of 58 tokens of a parked conversation after the long request and returned
  the same `cobalt` (`reviewed-bolt-greedy.json`). Repeating a discarded prompt also reproduced
  its original tokens, checking reset as well as restoration.
- Official Bolt, raw greedy: 4 prompts, 59 generated steps, exact match to the reference on every
  token. Teacher-forced: top-1 identical 59/59; the largest absolute probability delta over the
  common top-32 was 0.0212738043, the mean per-step max delta 0.0041537656, and the minimum
  top-32 overlap 29/32. A 3,245-token prompt produced 9 tokens, exact reference output
  `cobalt-5731` plus end-of-stream; time to first token 153.821 s (~21.10 prompt tokens/s).
- VRAM reserve: 512 → 768 → 512 MiB resized the cache from 410 to 252 to 403 experts, and the
  arithmetic over 16 tokens was unchanged. Process GPU memory sampled at 11,396 MiB.
- Low-RAM expert pool, 1,024 MiB: 3 requests, 43 tokens, exact. Two runs exist: `final-lowram.json`
  was run with `STRATA_FREEZE_EXPERTS=0`, which the code still treats as frozen because it keys on
  the variable's presence; `final-lowram-dynamic.json` ran with the variable unset and exercises
  expert promotion and the pool. Both pass.
- Load: 27.8 s, 410 of 13,824 experts cached, about 0.7 GiB of expert cache. The first cold
  raw-0 request had a 55.296 s time to first token while faulting expert pages from the HDD;
  subsequent short requests had 0.316 to 0.406 s time to first token. Warm short-request decode
  rates were about 20–23 tokens/s. These are measurements of this configuration, including
  serial prefill, rather than a performance comparison with the reference engine.
- The Metal build compiles the same `strata-poly` engine source. `nemotron_h_moe` is refused
  there by an explicit factory guard, and the two kernels it would need (`relu2_rows`,
  `mamba2_step`) throw `unported` rather than computing something else. The Metal build was not
  compiled or run here (no toolchain on the validation host); no Metal support is claimed. HIP and
  SYCL were also not built or run.

## Reproducible build and tests

```sh
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda-12.8/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_GGML_DIR=/path/to/pinned/llama.cpp -DSTRATA_BUILD_TESTS=ON
cmake --build build-cuda -j4 --target strata-poly poly_nemotron_route_test \
  poly_nemotron_meta_test poly_gguf_array_test poly_cpu_experts_relu2_test poly_workers_test
ctest --test-dir build-cuda -R 'poly_(nemotron_route|nemotron_meta|gguf_array|cpu_experts_relu2|workers)_test'
```

Replace `STRATA_GGML_DIR` with the path to a llama.cpp checkout at commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, which supplies the engine's ggml dependency.
The compiler paths and SM 86 match the validation host; select the CUDA architecture for your card.

Host tests run without a card; the GPU tests below need the CUDA target set and a card:

```sh
set -e
cmake --build build-cuda -j4 --target poly_relu2_test poly_conv4_bias_test \
  poly_mamba2_test poly_nemotron_scale_test poly_route_test poly_nvfp4_mmvq_test
for test in poly_relu2_test poly_conv4_bias_test poly_mamba2_test poly_nemotron_scale_test poly_route_test \
  poly_nvfp4_mmvq_test
do
  compute-sanitizer --tool memcheck --error-exitcode 99 "build-cuda/$test"
done
```

A separate CPU-only tree (`-DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON`) builds the three
pure-host tests without a CUDA toolchain. The tested runtime artifact's SHA-256 was
`4c4fcc2ff1c976255dad04c0e3371e3554614d99afd5c84fbaed143255146bd1`; local rebuilds can differ.

For full-model parity, use identical prompt token IDs, greedy sampling and frozen experts on
both PolyStrata builds. The existing `tools/qwen35/q36_solo_gate.py` and `q36_cmp_gate.py`
exercise the native protocol. For reference probabilities, `strata-poly --tokens ... --follow ...
--top-probs 32` evaluates the same generated history as llama.cpp. The reference server used
`--ctx-size 4096 --parallel 1 --gpu-layers 999 --cpu-moe --threads 13 --batch-size 128
--ubatch-size 128 --flash-attn on --no-warmup --fit off` and temperature 0. Preserve model,
source and binary hashes alongside the prompt IDs and results for each integration.

## Known gaps

- **Full Q4** of this architecture: not run; only the official IQ2_XXS + NVFP4 pack was exercised.
- **Full 32K / 128K context and long prompts**: a 30,000-token prompt into the 128K context ran
  correctly on 2026-11-10 (P1 measured below), taking the measured long request from 3,245 to 30K
  tokens. A context filled to 128K, and the KV-growth path that shrinks the expert cache once KV
  presses on it, are still unexercised.
- **HIP / SYCL**: no AMD or Intel GPU backend was built or run. The CUDA build is the only one
  exercised. The `sycl/` project's build lists no `src/poly` source at all, so `strata-poly` — and this
  architecture with it — does not exist there; Metal, by contrast, has the poly engine and refuses
  `nemotron_h_moe` by an explicit factory guard.
- **Metal**: the backend refuses this architecture (see above); it was not built or run.
- **Other full-model regressions**: GLM and other existing families were not run end to end;
  the full-model regression check covered Qwen3.6.
- **`moe_rows` / `moe_card`** stay GLU-shaped and are disabled for ReLU²/latent layers.
- **Serial admission only**: `--prompt-batch 1024` is clamped to 1 even when passed explicitly;
  `--prompt-batch 2`, `--window 2` and `--batch 1` are refused before allocation.

## Rolling integration

- Branch `bolt-backport` from the then-current `main` (`4de79331`); only the reviewed source
  commit `cc72c9f5` (source head `e59887c1`) is ported. The source's later Maxwell and
  unrelated-model commits are not part of this port.
- An update never lands on the accepted baseline directly. It goes through a staging branch
  (`bolt-backport-staging`) created from that baseline; the new upstream base is merged there
  first, then only newly reviewed Nemotron source changes are adapted —
  `git diff <old-source-SHA>..<new-source-SHA> -- <ported paths>` reviewed against the recorded
  scope — keeping the changes to the shared engine, kernels and GGUF reader no larger than the
  change requires. Conflicts are resolved by hand at that point: mechanically conflict-free
  updates are not promised. Only the new Nemotron changes since the last-reviewed source head
  (`e59887c1`) are ported; the whole initial port is not replayed and not all of the Tesla
  source is merged.
- Acceptance gates before the staging branch replaces the baseline: the host and GPU tests above,
  official-file reference/state/cache checks, and same-day existing-model regressions against
  the accepted baseline. Each update's old and new source
  SHAs and the new upstream base SHA go into the ledger below. Promote an accepted staging branch
  by merge or fast-forward; no force-push.
- Ledger: source SHA `cc72c9f5cec709ad66d881ea28a0de362c573fe3`, source head
  `e59887c1271b07b741669a327fe101102de69647`, upstream base
  `4de79331b49e553406860470fae96b5256e2e50`, common ancestor
  `dc2080ba16d7f9f644cd01c3d5ff3237b2ae0f85`. Local work on top: the ReLU² up dedup (`c74aad49`,
  2026-11-10) and the native NVFP4 compute (the same day) each went through `bolt-backport-staging`,
  passed the gates above and were promoted by fast-forward. Target MIT LICENSE is preserved;
  attribution to
  justlovett0 and the source repository
  (https://github.com/justlovett0/PolyStrata-TeslaM40) stays. Current fork
  https://github.com/skygunner/PolyStrata, upstream https://github.com/simplexai-labs/PolyStrata.
- Future `bolt-dev` branches start from an accepted `bolt-backport` baseline and merge accepted
  updates from it; speculative decoding and multimodal work stay there. Maxwell compatibility
  remains outside this fork's backport scope. No speculative or multimodal scaffolding is added now.
- For each delegated implementation round, record the ACP session and task, await its correlated
  terminal response, then inspect the actual patch and run independent acceptance. A normal
  `end_turn` confirms delivery, not correctness. Keep the supervisor active through handoff;
  do not leave an uncollected background job as the next integration's starting point.
- A PolyStrata upstream PR needs the remaining backend builds and review gates first. A native
  Strata port is a separate decision after this backend is established, with measured scheduling
  or prefill benefits to justify adapting its different execution path.

## Local fork operation

`/mnt/ssassd/PolyStrata` stays on `main` as the upstream installation; `/home/vincenty/Projects/PolyStrata`
is this fork on `bolt-backport`. The local artifacts below are gitignored and are not force-added:

- `run-strata.sh` (executable) — foreground launcher using this fork's `.venv/bin/python`,
  `serve/server.py` and `engine-poly/strata-poly`. It resolves its own directory, cds to it,
  runs the server with `--engine strata --config strata-nemotron-bolt-24gb.json --port 8081`,
  forwards server args after that default, and handles Ctrl-C normally. There is no `--stop`
  option and it never controls a service.
- `strata-nemotron-bolt-24gb.json` — host `127.0.0.1` in the JSON; the port is CLI-only
  (`server.py` default 8095 ignores `cfg.port`) and is set by the launcher, overridable with
  `--port`.
- `.venv` — a separate Python environment with the repository requirements installed.
- `engine-poly/strata-poly` — a copy of the validated binary, not a rebuild. After C++ changes
  rebuild and update the copy; a running binary is not changed by source edits. Since the NVFP4
  acceptance above (2026-11-10) the copy is the P1 build, SHA-256
  `e2da05d50c3a6cffae6dfb2742e2a0b7d72674891a3e48e0a76bc59d51b9e7ee`
  (the P0 build before it was `943c47a1…`).

If `polystrata.service` (note the r) is active, the launcher refuses and tells the user to run
`systemctl --user stop polystrata.service` first and `systemctl --user start polystrata.service`
after testing. The launcher itself never starts or stops it.

Accepted HTTP run, from `/tmp`: UI 200, health and model identity correct, `STRATA_FREEZE_EXPERTS`
unset even when inherited as `0`, two-turn chat `READY` then `cobalt` (27 prompt tokens, then 58
with 29 cached), launcher exit 0, upstream restored afterwards. Evidence:
`launcher/parent-http-acceptance.json`, `launcher/selfcheck.sh`,
`launcher/upstream-restored.json`. The first cold request took 87.14 s and the second
1.63 s; these are two distinct requests, not a throughput benchmark or a speedup claim. The
validated baseline config is context 4096 / max reply 2048 / `fit_max_tokens` true; later
128000 / 65536 settings are not the acceptance baseline.

## Memory audit, 2026-11-10

Measured or computed on the official pack; where a number is arithmetic rather than a measurement it
says so.

- 80 NVFP4 SSM projections, 40 input + 40 output; 4,383,047,680 elements. Packed on disk the block
  is 36 bytes per 64 elements, so the payload is
  `4,383,047,680 * 36 / 64 / 2**30 = 2.296142578 GiB`. Until 2026-11-10 `Engine::upload` had no
  native NVFP4 path and expanded them to BF16, 8.1640625 GiB; with type 40 in `native_mmvq` the
  packed bytes are what the card reads and the 5.867919921875 GiB difference is released (measured
  below).
- Other non-expert tensor payload, excluding the token embedding, is about 1.79 GiB. It is an
  estimate that includes tiny F32 CPU biases, so it is not an exact measured GPU allocation.
- ReLU² experts duplicate the up matrix. `load_nemotron_moe` aliases the same pointer and size for
  gate and up, `Layer::gu2` sums both, and `fill_expert_cache` copies twice. With equal-size matrices
  the current payload is 50% excess over the ideal up+down; the theoretical reduction of the current
  payload is 1/3, and the theoretical capacity gain for a fixed expert mix is 1.5x, ignoring
  padding. This is a calculation, not a measured capacity or performance figure. Fixed later the same
  day; the measured numbers are below.
- 40 MoE layers contain 13,824 actual expert instances:
  `12 * 240 + 6 * 304 + 9 * 368 + 10 * 432 + 3 * 496 = 13,824`.
  The maximum per-layer count, 496, determines the padded expert-table width; 13,824 is the
  actual sum across layers. 392 resident of 13,824 = 2.836% is the resident-capacity fraction,
  not a cache-hit percentage.
- Observations: at 4096 context the acceptance run cached 410 of 13,824 experts, about 0.7 GiB; a
  later 128000-context startup cached 392, about 0.7 GiB, with a 512 MiB reserve and 13 threads. The
  GPU process sampled at 11,392 MiB with about 490 MiB free in one snapshot. Desktop and driver load
  are dynamic, so the 18-expert difference between those two runs is not attributed to context
  alone. The KV cache grows on demand. A 128000-context startup plus a short request is **not** a
  full 128K correctness or stress validation.

- Slot arithmetic checked against the measured caches: from the file's own tensors — IQ2_XXS carries
  66 bytes per 256 values, the latent width is 1024, per-layer FFN widths run 1536 to 2560, and a slot
  pads to lcm(66, 8) plus the tail — the average three-matrix slot is 1.765 MiB and all 13,824 slots are
  23.83 GiB; storing up once gives 1.177 MiB and 15.89 GiB. The measured caches agree with this
  arithmetic: about 0.7 GiB held 392 to 410 experts, and 0.7 GiB divided by the average slot says
  about 406.
- Attention KV is cheap on this file: 8 attention layers of 32 query heads, 2 KV heads and head width
  128 hold 8 KiB per token, so a full 64K context carries 0.50 GiB of KV and a full 128K carries
  1.00 GiB. The Mamba state is fixed at 164.7 MiB per sequence. This is arithmetic on the file's
  shapes, not a measurement.
- Projection, arithmetic only and to be measured after the work: with the up matrix stored once and
  the NVFP4 projections kept packed, the cache would start near 6.5 GiB (about 5,900 experts, 43% of
  13,824), hold about 5.9 GiB (5,300) at a full 64K context and about 5.4 GiB (4,850, 35%) at a full
  128K — still short of every expert (15.89 GiB after dedup) on a 12 GiB card.

Known gaps are updated accordingly: no full 32K/128K runs exist despite the successful 128000
startup and short request. The untested backend, Q4 and other-model scopes above still stand.

## Prioritized improvement plan

Proposed work only. Nothing here is implemented, and no speed is promised.

- **P0 — deduplicate the ReLU² up matrix. Done 2026-11-10** (measured above: 614 experts at 4096
  context where 410 fit before, same 0.7 GiB). The engine derives every slot offset from `gu_bytes`,
  `u_bytes` and `gu2()`, and the ReLU² compute reads only the gate half of a slot (the up launch is
  skipped for ReLU²), so the change was one assignment in `load_nemotron_moe` — store the single up
  matrix as the gate and leave the up half empty — plus guards on the copy sites that would move an
  empty up.
- **P1 — native compressed NVFP4 compute. Done 2026-11-10** (measured below: 5,806 experts where 614
  fit before, and 68 against 22 tokens a second at decode). `vec_dot_nvfp4_q8_1` in
  `src/kernels/cuda/iq_kernels.cu` already dotted an NVFP4 block with a Q8_1 row on the GPU (written
  for DeepSeek's NVFP4 experts); `native_mmvq.cu` runs the same expression on the Q2_0 kernel's shape
  (a 64-value block a step) with type 40 in the three dispatch tables, a column at a time so a wider
  call gives each column the bits its own call gives. The BF16 expansion remains the fallback (and any
  other backend without the kernel), `lin()` already dispatched by type, CUDA and HIP compile the one
  file, and the SYCL build has no poly engine to owe (Known gaps).
- **P2 — batched/chunked Mamba prefill.** Needs numerical state proof and proper ReLU²/latent
  batched kernels; the current GLU-shaped `moe_rows`/`moe_card` stay disabled. A cold-I/O SSD
  comparison only after measurement; no automatic model move.
- **P3 — bolt-dev speculation.** Requires a rollback/checkpoint and a draft design; images require an
  encoder, projector and template validated independently. Keep the branch boundary: the accepted
  `bolt-backport` baseline versus experimental `bolt-dev`. PolyStrata upstream first; Strata is a
  separate architectural/value assessment, with backend CI gaps stated explicitly.

Every performance acceptance compares the same GPU, context, prompt token IDs, model SHA and
reserve against a same-day baseline, labels frozen versus dynamic and cold versus warm, and records
TTFT, prompt and decode rates, GPU and RAM, and cache hits. No speculative percentages are offered
as promises. The rolling-integration process above is reused rather than a new process or framework.

## P0 measured, 2026-11-10

The up-matrix dedup landed (staging commit `c74aad49`, promoted to `bolt-backport` the same day):
`load_nemotron_moe` leaves the up half of a ReLU² slot empty (`u_bytes` 0) and the seven copy sites
skip an empty up; the gate and down bytes are unchanged. Measured on the same card, file and reserve
as the validation above (4096 context, 512 MiB, 13 threads), against the previous accepted binary
(SHA-256 `4c4fcc2f…`, run the same day under identical invocations). Raw logs and the A/B driver are
in `/mnt/storage/PolyStrata-bolt-staging/acceptance-p0/`.

- Load cached **614 of 13,824 experts in 0.7 GiB** (was 410; the arithmetic above said about 406 for
  the three-matrix slot and 609 after dedup). A 128000-context startup cached **588** (was 392).
  Warm loads took 7 to 8 s.
- Greedy A/B, frozen cache, the four recorded raw prompts plus a repeat: five of six answers
  bit-identical to the old binary; the sixth (raw1) flipped its second token with the larger cache.
  With the cache shrunk to 408 experts the new binary reproduced the old answer bit for bit, and
  all-CPU runs (`--vram-reserve-mib 2688`, 0 experts cached) were bit-identical between the
  binaries: the flip is the GPU/CPU partition moving with the cache — the same class of difference
  as the probability deltas above, not a change in any expert's math. Each binary reproduced its
  own output exactly across reruns.
- The low-RAM pool path (1024 MiB, cache dynamic) answered the recorded prompts bit-identically to
  the old binary, exercising the pool's up|down layout.
- Qwen3.6 UD-Q4_K_XL, same day, three of the recorded ten prompts: bit-identical to the old binary.
- The five GPU tests under compute-sanitizer: 0 errors. The host and CPU-tree tests pass as before.
- Not redone here: the 3,245-token long prompt, the HTTP loopback suite and the VRAM-resize cycle;
  the long-context runs at 64K/128K belong to P1's acceptance plan above. HIP was not built (no
  toolchain on this machine); the change is host-side only.

## P1 measured, 2026-11-10

Native NVFP4 compute landed (staging, promoted the same day): `native_mmvq.cu` gained `NVFP4Block`,
`nvfp4_q8_dot` (the `vec_dot_nvfp4_q8_1` expression of `iq_kernels.cu`, the same tables and UE4M3
decode) and `native_nvfp4_mmvq_kernel` on the Q2_0 kernel's shape, plus type 40 in
`native_mmvq_supported`, `native_mmvq_weight_bytes` and `native_mmvq`. The engine changes with it:
`upload` keeps the 36-byte blocks as the file has them (the BF16 expansion stays as any other
backend's fallback) and `lin()` dispatches them to the new kernel. `poly_nvfp4_mmvq_test`
(`src/poly/nvfp4_mmvq_test.cu`) checks the kernel against ggml's own dequant of the same blocks and
the engine's own Q8_1 values: relative error 7.3e-8 over the three shapes (the projections' two
widths and a small-K one), and a three-column call gives each column the bits its own call gives.
All measured on the same card, file and settings as above against the P0 build (`943c47a1…`, the
accepted baseline of the day). Raw logs and the drivers are in
`/mnt/storage/PolyStrata-bolt-staging/acceptance-p0/` (`ab-tokens-p1.json`, `probs-p0/p1.txt`).

- Startup residency: **5,806 of 13,824 experts (6.6 GiB)** at 4096 context, 5,794 at 64K and 5,780 at
  128K — against 614 / 588 / 5780's predecessor 392 before P0. The arithmetic above said about 5,900.
- Greedy A/B against the P0 build, frozen cache, the four recorded raw prompts: **bit-identical on
  every token**; the teacher-forced top-32 over raw0's eleven steps: top-1 identical 11/11, the
  largest probability delta on a common token 0.019 (the port's own acceptance band was 0.0213
  against llama.cpp), minimum top-32 overlap 31/32 (that band: 29/32). The two unfrozen low-RAM
  prompts: one bit-identical, one diverged at token 2 with the much larger cache — the same
  partition effect P0's acceptance documented, not a change in any expert's or projection's math.
- Decode, 64 tokens a run, three runs each, same prompt: **21.95–22.30 tokens/s (P0) against
  67.7–68.4 (P1)**; prefill of the 5-token prompt 15.2 against 34.7. The expert hit rate rose from
  26.6% to 67.5% of expert picks, the router wait fell from 35.8 to 8.7 ms/token and the CPU experts
  from about 8.8 to 4.7 ms/token: the released 5.87 GiB went to experts, and the ssm reads shrank
  with it.
- A 30,000-token prompt into the 128K context: prefill at 43.4 tokens/s (12m35s), 96.6% of expert
  picks from VRAM, 16 tokens answered, no error; its output matched the P0 build's answer to the
  same prompt token for token (the long-context A/B). The KV-growth path that shrinks the expert
  cache was still not exercised: 30K of context is 0.23 GiB of KV beside a 6.5 GiB cache.
- Qwen3.6, same day, three of the recorded prompts: bit-identical to the P0 build (the change is one
  case in a dispatch the Q4/Q8 files never take).
- The six GPU tests under compute-sanitizer: 0 errors. HIP was not built (no toolchain here); the
  dot's only device helpers, `table16_lookup` and `STRATA_DP4A`, are the MXFP4 dot's, which HIP
  already compiles.
