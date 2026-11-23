# Qwen3.8 Flash-Next EXL3

A Gufo extension for turboderp's EXL3 checkpoints of Qwen3.8 Flash-Next on
Linux x86-64 / Strix Halo `gfx1151`. It builds on the Flash-Next runtime of
Gufo's `qwen38_flash_next` extension instead of copying it.

**Status: format and kernels only.** This repository holds the EXL3 trellis
format, the checkpoint reader, the scalar host reference and the HIP linears,
with their tests. It does not serve a model yet. Serving waits for the
Flash-Next Stages and WeightSource interface; see
[not ported yet](#not-ported-yet).

Gufo requires every extension to declare a provider, so this one declares a
placeholder, `flash-next-exl3`, which registers as unavailable. Its keys,
`safetensors:qwen4_exp` and `safetensors:Qwen4ExpForConditionalGeneration`,
match any Flash-Next safetensors checkpoint, BF16 included, not only EXL3
ones. They are placeholders for the real providers. The unavailable reason
shows only in `gufo diagnose --section extensions`; loading a checkpoint
directory fails with "no text provider claims model directory".

Maintainer: Federico Izzo ([fedeizzo](https://github.com/fedeizzo)).

## Supported Gufo revisions

| Extension | Gufo | Notes |
| --- | --- | --- |
| 0.1.0 | `cab68390c0ddb7e944cd363287b836a54237c48a` ([#1360](https://github.com/gufo-org/gufo/pull/1360)) | Needs `depends` linking and the `gufo::qwen38_flash_next_hip` export |

The extension depends on `qwen38_flash_next` (`"depends"` in
[extension.json](extension.json)). Links to that extension's export carry no
compatibility promise, so pin the Gufo revision you tested.

## Checkpoints

| Model ID | Checkpoint | Download | Next-token loss, Q4 GGUF = 1.417 |
| --- | --- | --- | --- |
| `flash-next-exl3-4bpw` | `4.05bpw_h6_ng6` | 108 GB | 1.412 |
| `flash-next-exl3-3bpw` | `3.05bpw_h5_ng5` | 86 GB | 1.426 |
| `flash-next-exl3-2bpw` | `2.05bpw_h4_ng4` | 64 GB | 1.439 |

These are the checkpoints the providers will serve, measured with Gufo's EXL3
branch. Each is a directory of safetensors files that contains the MTP head.
Its n-gram table (39, 33 and 26 GB) never enters GPU memory: rows are read from
disk per token. The checkpoints are revisions of
[turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)
(4.05 bpw at revision `55a732e0`); their weights keep the Qwen community
license (`qwen-community-1.0`) and are downloaded separately.
[EXL3 notes](docs/EXL3.md) has the format, kernels, quality and performance.

## Layout

| Path | Contents |
| --- | --- |
| `exl3/format.hpp` | mul1 codebook, per-lane tile window and n-gram row codec, shared by host and device |
| `exl3/pack.{hpp,cpp}` | Read-only view of an EXL3 checkpoint directory: mapped safetensors, tensor index, JSON configuration, cache identity |
| `exl3/reference.{hpp,cpp}` | Scalar host reference of EXL3 linears and n-gram rows |
| `exl3/rocm/exl3.{hpp,hip.cpp}` | EXL3 linears for gfx11 wave32: decode GEMV, expert GEMM, Q8_0 expansion, repacking |
| `extension.cpp` | Registration and the unavailable placeholder provider |
| `tests/exl3_format_test.cpp` | Format contracts against Crucible's NumPy reference and ExLlamaV3's n-gram codec; no GPU |
| `tests/exl3_pack_test.cpp` | Checkpoint reader on tiny synthetic safetensors directories: detection, config queries, identity inputs, malformed headers; no GPU |
| `tests/exl3_ops_test.cpp` | Every kernel against the host reference for K2 to K6; GPU, labelled `slow` (about 35 s) |
| `tests/exl3_gemv_bench.cpp` | Decode and prompt-pass linear timings on random weights; run by hand |

The code comes from Gufo's `feat/qwen38-flash-next-exl3` branch, commits
`197ff374` (EXL3 checkpoints) and `bb5dff44` (faster decode and prompt
passes), rebased on Gufo `main`. Git history was not carried over. The C++
namespaces are still `gufo::models::qwen38_flash_next::{exl3,rocm}`. Changes
from the branch, besides include paths and header guards:

- `exl3/pack.{hpp,cpp}`: `Pack::Open` rejects tensor offsets and shapes whose
  64-bit arithmetic wraps (`PackTensor::CheckedBytes`, and an end bound checked
  against the bytes after the header).
- `exl3/rocm/exl3.hpp`: attribution of the front tile order to Crucible's
  `FrontTile`.
- `tests/exl3_pack_test.cpp` is new.

`Pack::Identity()` hashes only `config.json` and the safetensors headers, not
tensor payloads: a checkpoint re-quantized to the same names, dtypes, shapes
and offsets gets the same identity. Fix this before serving uses it as a cache
key.

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
  qwen38_flash_next_exl3_ops_test \
  qwen38_flash_next_exl3_gemv_bench
ctest --test-dir build/exl3 -R '^qwen38_flash_next_exl3\.' --output-on-failure
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
`clang-format -i exl3/*.hpp exl3/*.cpp exl3/rocm/* tests/*.cpp extension.cpp`.

## Not ported yet

These parts of the EXL3 branch stay in Gufo's branch until Flash-Next
exposes its layered library (RFC step 5: `gufo::qwen_flash_next` with
`Stages`, `WeightSource` and opaque `Layer`/`Tensor` handles).

| Branch code | What it does | Interface it needs |
| --- | --- | --- |
| `exl3/bind.cpp` (`Config::FromExl3`, `ModelWeights::BindExl3`, `MtpWeights::BindExl3`) | Builds the Flash-Next configuration and weight descriptors from a pack; converts BF16/F16 tensors to Q8_0 | `WeightSource::Bind()` returning descriptors, host payloads and a per-tensor `format_state`; a public Flash-Next config type |
| `weights.hpp` `Exl3Ref`, `Exl3NgramRef`, `TensorRef::exl3`, `ple_exl3` | EXL3 fields on Flash-Next weight descriptors | `Tensor::format_state`, owned by the extension |
| `kernels/rocm/device_model.{hpp,cpp}` `UploadExl3`, `DeviceTensor::exl3_*` | Uploads trellis tiles, Hadamard signs and front-order tiles | `WeightSource::Upload(tensor, device)` |
| `kernels/rocm/executor.{hpp,cpp}` `Exl3Project`, `Exl3RowGroup`, `Exl3MoeRows`, `Exl3PromptLayer`, `Exl3ExpandLayer`, `exl3_scratch_` | Dispatches EXL3 linears in decode, verification and prompt passes | `Stages::Linear`, `Stages::LinearGroup`, `Stages::MoeFfn`, `Stages::ScratchBytes`, with HIP graph capture kept working |
| `ngram.{hpp,cpp}` trellis rows | Decodes n-gram rows of the on-disk table | An n-gram row source hook in Flash-Next's n-gram embedding |
| `engine.{hpp,cpp}`, `runner.cpp` pack handling | Opens the pack instead of a GGUF, larger prompt chunk, cache identity, checkpoint MTP head | `WeightSource::Probe`, `Identity`, `Tokenizer`, `PreferredPrefillChunk`; nullable GGUF reader in the text loader |
| `probe.{hpp,cpp}` `ProbeFlashNextExl3` | Accepts an EXL3 checkpoint directory | `WeightSource::Probe(path)` and model-directory probing in Gufo's loader |
| `src/models/qwen38_flash_next_exl3_{2,3,4}bpw/` | One provider extension per checkpoint, with `models` entries and `models.lock.json` | `qwen_flash_next::Create(Options, Stages, WeightSource)`; providers then replace the placeholder here |
| `tests/.../exl3_bind_probe.cpp`, `exl3_generate_probe.cpp` | Bind and end-to-end generation probes on real checkpoints | The bind and create paths above |
| Tokenizer from `tokenizer.json` | Checkpoint tokenizer | `QwenTokenizer::CreateFromTokenizerJson` in the Qwen family layer |

## TODO

- Raise `api_version` in [extension.json](extension.json) to the extension
  API revision of gufo-org/gufo#1360's change record once #1360 merges.
- Once #1360 merges, re-pin [flake.nix](flake.nix), `flake.lock` and the
  supported revision above to its squash commit on Gufo `main`.
- Move the code to an extension-owned namespace in a follow-up PR.
- Hash the tensor payloads (or a checkpoint revision) into `Pack::Identity()`
  before serving uses it as a cache key.

## License

The extension source is released under the [MIT License](LICENSE). It adapts
code from Crucible (Apache-2.0) and follows ExLlamaV3's format (MIT); see
[third-party notices](THIRD_PARTY_NOTICES.md). Model weights are not part of
this repository and keep their own licenses.
