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
  architecture has been exercised. NVFP4 `ssm_in` / `ssm_out` are dequantized to BF16 at load and
  their F32 `.scale` is applied to the projection result, as llama.cpp's
  `build_lora_mm(..., ssm_in_s)` does. The source supplied the architecture implementation;
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
  poly_mamba2_test poly_nemotron_scale_test poly_route_test
for test in poly_relu2_test poly_conv4_bias_test poly_mamba2_test poly_nemotron_scale_test poly_route_test
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
- **32K / 128K context and long prompts**: not run. The measured long request was 3,245 tokens.
- **HIP / SYCL**: no AMD or Intel GPU backend was built or run. The CUDA build is the only one
  exercised.
- **Metal**: the backend refuses this architecture (see above); it was not built or run.
- **Other full-model regressions**: GLM and other existing families were not run end to end;
  the full-model regression check covered Qwen3.6.
- **Expert slots duplicate the single up matrix** (2× that matrix), which reduces cache capacity
  for a given VRAM reserve. It is a known limitation of the port, not an optimization.
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
  `4de79331b49e553406860470fa3e96b5256e2e50`, common ancestor
  `dc2080ba16d7f9f644cd01c3d5ff3237b2ae0f85`. Target MIT LICENSE is preserved; attribution to
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
