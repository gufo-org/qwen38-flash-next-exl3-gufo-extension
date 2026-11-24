# Qwen3.8 Flash-Next EXL3

A Gufo extension for turboderp's EXL3 checkpoints of Qwen3.8 Flash-Next on
Linux x86-64 / Strix Halo `gfx1151`. It builds on the Flash-Next runtime of
Gufo's `qwen38_flash_next` extension instead of copying it.

It serves the three checkpoints, `flash-next-exl3-4bpw`,
`flash-next-exl3-3bpw` and `flash-next-exl3-2bpw`, through Flash-Next's
`Stages` and `WeightSource` interface: Flash-Next keeps the pipeline (passes,
KV, recurrent state, MTP, rollback, snapshots, sampling), and this extension
supplies the checkpoint binding, the trellis upload and the stages that run
EXL3 linears. `--speculative mtp` drafts from the checkpoint's own MTP block
and needs no `--mtp-model`.

```sh
gufo model install flash-next-exl3-4bpw
gufo serve llm --model flash-next-exl3-4bpw --speculative mtp --context 32768
gufo prompt -m /path/to/4.05bpw_h6_ng6 -p "Hello" --think off
```

The providers' keys, `safetensors:qwen4_exp` and
`safetensors:Qwen4ExpForConditionalGeneration`, are the ones Gufo's EXL3
branch used, and Gufo has no narrower key. They are broad: they match any
Flash-Next safetensors checkpoint, BF16 included, and each provider's probe
then accepts only EXL3 directories of its bit width. Gufo refuses to install
a loadable module (`gufo extension install`) whose text provider shares a key
with a registered provider, and CMake discovery stops when a shipped and an
unshipped extension share a key. A future BF16 or FP8 Flash-Next safetensors
extension would therefore conflict with this one until the keys can be
narrowed.

Maintainer: Federico Izzo ([fedeizzo](https://github.com/fedeizzo)).

## Supported Gufo revisions

| Extension | Gufo | Notes |
| --- | --- | --- |
| 0.1.0 | `d3a7bab1563e7266c52c426a149db00841b6d4d6` | Needs Flash-Next `Stages`/`WeightSource` ([#1364](https://github.com/gufo-org/gufo/pull/1364), merged), `depends` linking and exports ([#1360](https://github.com/gufo-org/gufo/pull/1360), merged), `QwenTokenizer::CreateFromTokenizerJson` ([#1359](https://github.com/gufo-org/gufo/pull/1359), merged), `gufo::qwen_family` ([#1361](https://github.com/gufo-org/gufo/pull/1361), merged) and the public Flash-Next headers and hooks ([#1372](https://github.com/gufo-org/gufo/pull/1372), merged) |

The extension depends on `qwen38_flash_next` (`"depends"` in
[extension.json](extension.json)). Links to that extension's export carry no
compatibility promise, so pin the Gufo revision you tested.

## Checkpoints

| Model ID | Checkpoint | Download | Next-token loss, Q4 GGUF = 1.417 |
| --- | --- | --- | --- |
| `flash-next-exl3-4bpw` | `4.05bpw_h6_ng6` | 108 GB | 1.412 |
| `flash-next-exl3-3bpw` | `3.05bpw_h5_ng5` | 86 GB | 1.426 |
| `flash-next-exl3-2bpw` | `2.05bpw_h4_ng4` | 64 GB | 1.439 |

These are the checkpoints the providers serve, measured with Gufo's EXL3
branch. Each is a directory of safetensors files that contains the MTP head.
Its n-gram table (39, 33 and 26 GB) never enters GPU memory: rows are read from
disk per token. The checkpoints are revisions of
[turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)
(4.05 bpw at revision `55a732e0`); their weights keep the Qwen community
license (`qwen-community-1.0`) and are downloaded separately.
[EXL3 notes](docs/EXL3.md) has the format, kernels, quality and performance.

## Model weights

[extension.json](extension.json) declares one model per checkpoint, with the
same IDs as its provider. [models.lock.json](models.lock.json) pins every file
by revision, size and SHA-256. The weights component takes the checkpoint's
safetensors, JSON, Jinja and text files and its `LICENSE`, without
`qbench_prompts.json` and `quantization_config.json`. The default `mmproj`
component is unsloth's `mmproj-BF16.gguf` (908 MB), which Flash-Next uses
for images.

```sh
gufo model list                          # the three models, size and install state
gufo model list --json                   # components, sources and bytes
gufo model install flash-next-exl3-2bpw  # weights and mmproj
gufo model verify flash-next-exl3-2bpw   # rehash the installed files
gufo model update flash-next-exl3-2bpw   # follow a new lock
gufo model remove flash-next-exl3-2bpw
```

`gufo model list` shows `available` in the sense of downloadable; `gufo
diagnose --section extensions` reports the providers.

`install` reuses a file from the Hugging Face cache (`$HF_HUB_CACHE`, else
`$HF_HOME/hub`) when `models--<owner>--<repo>/blobs/<sha256>` exists for the
component's repository, with the SHA-256 from `models.lock.json`. To install a
checkpoint you already downloaded without fetching it again, hard-link every
file of every component into its repository's `blobs` directory, on the same
file system. Name each link by the file's locked SHA-256. A real Hugging Face
cache names small non-LFS files (JSON, Jinja, text, `LICENSE`) by their Git
SHA-1 instead, so `install` downloads those files again.

```sh
# Weights: every file of the model's "weights" entry in models.lock.json.
mkdir -p "$HF_HUB_CACHE/models--turboderp--Qwen3.8-Flash-Next-exl3/blobs"
ln /path/to/2.05bpw_h4_ng4/<path> \
  "$HF_HUB_CACHE/models--turboderp--Qwen3.8-Flash-Next-exl3/blobs/<sha256>"
# Image projector: the "mmproj" entry, from another repository.
mkdir -p "$HF_HUB_CACHE/models--unsloth--Qwen3.8-Flash-Next-GGUF/blobs"
ln /path/to/mmproj-BF16.gguf \
  "$HF_HUB_CACHE/models--unsloth--Qwen3.8-Flash-Next-GGUF/blobs/<sha256>"
```

`install` still hashes each file before using it. `remove` deletes only the
store's links, and the original files stay.

## Layout

| Path | Contents |
| --- | --- |
| `exl3/format.hpp` | mul1 codebook, per-lane tile window and n-gram row codec, shared by host and device |
| `exl3/pack.{hpp,cpp}` | Read-only view of an EXL3 checkpoint directory: mapped safetensors, tensor index, JSON configuration, cache identity |
| `exl3/reference.{hpp,cpp}` | Scalar host reference of EXL3 linears and n-gram rows |
| `exl3/rocm/exl3.{hpp,hip.cpp}` | EXL3 linears for gfx11 wave32: decode GEMV, expert GEMM, Q8_0 expansion, repacking |
| `extension.cpp` | Extension entry; the providers register from `flash_next/provider.cpp` |
| `flash_next/bind.{hpp,cpp}` | Flash-Next configuration (checked by `Config::Validated`) and weight descriptors from a pack: trellis linears carry an `Exl3Ref` as `format_state`, norms are folded, DeltaNet heads permuted, hyper-connection and PLE projections quantized to Q8_0 |
| `flash_next/weight_source.{hpp,cpp}` | `WeightSource`: probe, native context, the checkpoint's own MTP block, bind, `tokenizer.json`, trellis n-gram rows (`NgramTable::OpenEncoded`), trellis upload and repack, Q8_0 draft head in the weight heap, cache identity, 8,192-row prompt passes |
| `flash_next/stages.{hpp,cpp}` | `Stages`: grouped trellis GEMVs for decode and verification, trellis GEMMs for prompt passes, the MoE FFN on both, and the per-layer Q8_0 expansion prompt passes run on (`PromptLayer`) |
| `flash_next/provider.cpp` | The three providers, told apart by the checkpoint's bits per weight, over Flash-Next's non-GGUF text runner |
| `models.lock.json` | Size and SHA-256 of every model file, from Gufo's EXL3 branch |
| `tests/exl3_format_test.cpp` | Format contracts against Crucible's NumPy reference and ExLlamaV3's n-gram codec; no GPU |
| `tests/exl3_pack_test.cpp` | Checkpoint reader on tiny synthetic safetensors directories: detection, config queries, identity inputs, malformed headers; no GPU |
| `tests/exl3_ops_test.cpp` | Every kernel against the host reference for K2 to K6; GPU, labelled `slow` (about 35 s) |
| `tests/exl3_gemv_bench.cpp` | Decode and prompt-pass linear timings on random weights; run by hand |
| `tests/exl3_bind_test.cpp` | Binds a real checkpoint of each bits class and checks the descriptors; no GPU, skips without `QWEN38_FLASH_NEXT_EXL3_MODEL` |
| `tests/exl3_generate_test.cpp` | Per bits class, through `Model::Load` with these stages and source: greedy AR answer, prompt against decode path, one MTP session and two batched MTP sessions (verification wider than the grouped route) against AR; GPU, skips without `QWEN38_FLASH_NEXT_EXL3_MODEL` |

The code comes from Gufo's `feat/qwen38-flash-next-exl3` branch, commits
`197ff374` (EXL3 checkpoints) and `bb5dff44` (faster decode and prompt
passes), rebased on Gufo `main`. Git history was not carried over. The code
lives in `gufo::ext::qwen38_flash_next_exl3` (kernels in its `rocm`
namespace). Changes from the branch, besides include paths and header guards:

- `exl3/pack.{hpp,cpp}`: `Pack::Open` rejects tensor offsets and shapes whose
  64-bit arithmetic wraps (`PackTensor::CheckedBytes`, and an end bound checked
  against the bytes after the header). `Pack::Identity()` also hashes the
  first and last 64 bytes of every tensor payload, so a checkpoint with any
  tensor re-quantized under the same layout gets another cache identity.
- `exl3/rocm/exl3.hpp`: attribution of the front tile order to Crucible's
  `FrontTile`.
- The branch's hooks in the Flash-Next executor, device model, engine and
  runner became `Stages` and `WeightSource` overrides (`flash_next/`). The
  prompt-pass shared expert still forks after the routed gate and up. The
  repack staging is freed by `UploadDone`, and the Q8_0 draft head is
  allocated in the weight heap during the upload.
- `--mtp-model` naming another path than the checkpoint is an error.
- `tests/exl3_pack_test.cpp` is new; the bind and generate probes became the
  model-gated `bind` and `generate` tests.

`exl3_ops_test` compares the fused SwiGLU group with Flash-Next's own `Swiglu`
kernel, so it includes `src/models/qwen38_flash_next/kernels/rocm/kernels.hpp`
through the `gufo::qwen38_flash_next_hip` export. That is a private Gufo
header; the test is the only user.

## Build and test

```sh
cmake --preset gpu-test -B build/exl3 \
  -DGUFO_EXTENSION_DIRS=/absolute/path/to/qwen38-flash-next-exl3-gufo-extension
cmake --build build/exl3 --parallel 4 --target \
  qwen38_flash_next_exl3_format_test qwen38_flash_next_exl3_pack_test \
  qwen38_flash_next_exl3_ops_test qwen38_flash_next_exl3_bind_test \
  qwen38_flash_next_exl3_generate_test qwen38_flash_next_exl3_gemv_bench
ctest --test-dir build/exl3 -R '^qwen38_flash_next_exl3\.' --output-on-failure
# The bind and generate tests skip unless a checkpoint is named:
QWEN38_FLASH_NEXT_EXL3_MODEL=/path/to/4.05bpw_h6_ng6:/path/to/3.05bpw_h5_ng5 \
  ctest --test-dir build/exl3 -R '^qwen38_flash_next_exl3\.(bind|generate)_' \
  --output-on-failure
build/exl3/qwen38_flash_next_exl3_gemv_bench
```

Run these from a Gufo checkout at a supported revision, inside its `nix
develop` shell or with the documented dependencies. `qwen38_flash_next_exl3.format`
and `qwen38_flash_next_exl3.pack` run on the CPU; `qwen38_flash_next_exl3.ops`
needs a `gfx1151` GPU.

With Nix, [flake.nix](flake.nix) pins Gufo with this extension added:

```sh
nix build
./result/bin/gufo diagnose --section extensions
```

Format C++ with Gufo's [.clang-format](.clang-format):
`clang-format -i exl3/*.hpp exl3/*.cpp exl3/rocm/* flash_next/* tests/*.cpp extension.cpp`.

## Flash-Next interface

The extension uses Flash-Next's experimental headers under
`<gufo/ext/families/qwen/models/flash_next/>` (no compatibility promise):
`Stages`, `WeightSource` (with `NativeContextLength`, `MtpPath` and
`UploadDone`), `Config::Validated`, `NgramTable::OpenEncoded` and the
non-GGUF `CreateQwenFlashNextTextRunner`. The `gufo::qwen38_flash_next`
exports still put private Gufo headers on the include path, and
`tests/exl3_ops_test.cpp` uses one (`kernels/rocm/kernels.hpp`).

## License

The extension source is released under the [MIT License](LICENSE). It adapts
code from Crucible (Apache-2.0) and follows ExLlamaV3's format (MIT); see
[third-party notices](THIRD_PARTY_NOTICES.md). Model weights are not part of
this repository and keep their own licenses.
