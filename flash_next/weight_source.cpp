#include "flash_next/weight_source.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <deque>
#include <filesystem>
#include <gufo/ext/families/qwen/models/flash_next/ngram.hpp>
#include <gufo/ext/families/qwen/text/tokenizer.hpp>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "exl3/format.hpp"
#include "exl3/pack.hpp"
#include "exl3/reference.hpp"
#include "exl3/rocm/exl3.hpp"
#include "flash_next/bind.hpp"
#include "flash_next/device_format.hpp"

namespace gufo::ext::qwen38_flash_next_exl3 {
namespace {

namespace fn = ::gufo::ext::qwen::flash_next;
using ::gufo::models::qwen38_flash_next::NgramTable;

/// Read past the end of every device tensor by the vectorized loads.
constexpr std::size_t kTailMargin = 4096;

class Exl3WeightSource final : public fn::WeightSource {
public:
  ~Exl3WeightSource() override { ReleaseStaging(); }

  [[nodiscard]] bool Probe(const std::string& model_path) const override {
    return IsFlashNextPack(model_path);
  }

  [[nodiscard]] std::uint32_t NativeContextLength(
      const std::string& model_path) const override {
    return PackContextLength(model_path);
  }

  /// The checkpoint carries its own MTP block. A requested draft model that
  /// names another path is passed on, so Open reports it.
  [[nodiscard]] std::string MtpPath(
      const std::string& model_path,
      const std::string& requested) const override {
    std::error_code ec;
    return requested.empty() ||
                   std::filesystem::equivalent(requested, model_path, ec)
               ? model_path
               : requested;
  }

  bool Open(const std::string& model_path, const std::string& mtp_path,
            std::string* error) override {
    // The checkpoint carries its own MTP block: a draft path may only name
    // the checkpoint itself.
    std::error_code ec;
    if (!mtp_path.empty() &&
        !std::filesystem::equivalent(mtp_path, model_path, ec)) {
      if (error != nullptr)
        *error =
            "an EXL3 checkpoint drafts from its own MTP block; omit "
            "--mtp-model or pass the checkpoint directory (" +
            model_path + "), not " + mtp_path;
      return false;
    }
    drafts_ = !mtp_path.empty();
    pack_ = Pack::Open(model_path, error);
    return pack_ != nullptr;
  }

  bool BindTrunk(ModelWeights* weights, std::string* error) override {
    auto bound =
        qwen38_flash_next_exl3::BindTrunk(*pack_, host_, &ngram_, error);
    if (!bound)
      return false;
    *weights = std::move(*bound);
    return true;
  }

  bool BindMtp(const Config& trunk, MtpWeights* mtp,
               std::string* error) override {
    std::string mtp_error;
    auto bound =
        qwen38_flash_next_exl3::BindMtp(*pack_, trunk, host_, &mtp_error);
    if (!bound) {
      if (error != nullptr)
        *error = mtp_error.empty() ? "the EXL3 checkpoint has no MTP block"
                                   : mtp_error;
      return false;
    }
    *mtp = std::move(*bound);
    return true;
  }

  std::unique_ptr<::gufo::tokenization::QwenTokenizer> Tokenizer(
      const Config& config, std::string* error) override {
    return ::gufo::tokenization::QwenTokenizer::CreateFromTokenizerJson(
        pack_->ReadText("tokenizer.json"), config.vocab_size, error);
  }

  std::unique_ptr<NgramTable> Ngram(const ModelWeights& weights,
                                    std::string* error) override {
    const Config& c = weights.config;
    const NgramRef& t = ngram_;
    std::vector<std::uint32_t> head_offsets(
        c.ple_head_offsets.begin(), c.ple_head_offsets.begin() + c.ple_heads);
    if (c.ple_head_dim != kNgramRowDim ||
        (t.k != 4 && t.k != 5 && t.k != 6 && t.k != 8) ||
        head_offsets.empty() ||
        t.head_bias.size() != head_offsets.size() * c.ple_head_dim ||
        t.file >= pack_->Regions().size()) {
      if (error != nullptr)
        *error = "unsupported n-gram trellis table";
      return nullptr;
    }
    const std::size_t row_bytes = 2 * NgramRowWords(t.k);
    // row[i] = mul1(state_i) * scale + head_bias[head][i] (exl3/format.hpp).
    auto decode = [k = t.k, row_bytes, bias = t.head_bias,
                   starts = std::move(head_offsets)](
                      const std::uint8_t* src, std::uint32_t row, float* dst) {
      const auto head = static_cast<std::size_t>(
          std::upper_bound(starts.begin(), starts.end(), row) - starts.begin() -
          1);
      std::uint16_t words[1 + kNgramRowDim / 2];
      std::memcpy(words, src, row_bytes);
      DecodeNgramRow(words, k, bias.data() + head * kNgramRowDim, dst);
    };
    return NgramTable::OpenEncoded(pack_->Regions()[t.file].file_descriptor,
                                   t.file_offset, t.rows, c.ple_head_dim,
                                   row_bytes, std::move(decode), error);
  }

  [[nodiscard]] std::span<const core::GgufMappedRegion> Regions(
      bool mtp) const override {
    // The MTP block reads the same checkpoint files as the trunk.
    (void)mtp;
    return pack_->Regions();
  }

  void UploadDone(bool measuring) override {
    // The repack staging serves one uploading walk.
    if (!measuring)
      ReleaseStaging();
  }

  fn::Tensor Upload(const TensorRef& t, const fn::UploadHint& hint,
                    fn::UploadContext& up) override;

  bool DraftHead(const fn::Tensor& output, fn::Tensor* head,
                 std::string* error) override;

  [[nodiscard]] std::string Identity() const override {
    return pack_->Identity();
  }
  [[nodiscard]] std::string ModelName() const override {
    return "Qwen3.8 Flash Next";
  }
  [[nodiscard]] std::string VisionPath(
      const std::string& model_path) const override {
    // The projector sidecar is searched inside the checkpoint directory.
    return (std::filesystem::path(model_path) / "config.json").string();
  }
  [[nodiscard]] std::uint32_t PreferredPrefillChunk() const override {
    return kPrefillChunkTokens;
  }

private:
  void ReleaseStaging() noexcept {
    if (staging_ != nullptr)
      (void)hipFree(staging_);
    staging_ = nullptr;
    staging_bytes_ = 0;
  }

  std::unique_ptr<Pack> pack_;
  HostTensors host_;
  NgramRef ngram_;
  std::deque<DeviceFormat> formats_;
  /// Device staging the stored payloads land in before the repack, outside
  /// the weight heap; freed by UploadDone, or with the source.
  void* staging_{nullptr};
  std::size_t staging_bytes_{0};
  bool drafts_{false};       ///< Opened with the MTP block.
  fn::Tensor draft_head_{};  ///< Q8_0 head in the weight heap (DraftHead).
};

fn::Tensor Exl3WeightSource::Upload(const TensorRef& t,
                                    const fn::UploadHint& hint,
                                    fn::UploadContext& up) {
  const Exl3Ref* ref_ptr = RefOf(t);
  if (ref_ptr == nullptr) {
    up.Fail("tensor " + std::string(t.name) + " carries no EXL3 state");
    return {};
  }
  const Exl3Ref& ref = *ref_ptr;
  const std::size_t n = ref.parts.size();
  const std::size_t in = ref.In();
  const std::size_t out = ref.Out();
  const std::size_t tile_bytes = ref.DeviceBytes() * n;
  void* tiles = nullptr;
  void* suh = nullptr;
  void* svh = nullptr;
  // With MTP, the head's Q8_0 expansion (DraftHead) lives in the weight
  // heap beside it, so the model's resident bytes count it.
  const bool draft_head = drafts_ && t.name == "lm_head" && n == 1;
  const std::size_t q8_bytes = out * (in / 32) * 34;
  void* q8 = nullptr;
  if (!up.Allocate(&tiles, tile_bytes + kTailMargin) ||
      !up.Allocate(&suh, n * in * 2 + kTailMargin) ||
      !up.Allocate(&svh, n * out * 2 + kTailMargin) ||
      (draft_head && !up.Allocate(&q8, q8_bytes + kTailMargin))) {
    up.Fail("device allocation failed for " + std::string(t.name));
    return {};
  }
  if (up.Measuring())
    return {};
  const auto fail = [&](std::string message) -> fn::Tensor {
    ReleaseStaging();
    up.Fail(message);
    return {};
  };
  // Staging: [n][stored tiles] | [n][src_in] suh | [n][src_out] svh |
  // offsets | block maps.
  const std::size_t src = ref.SourceBytes();
  const std::size_t suh_at = n * src;
  const std::size_t svh_at = suh_at + n * ref.src_in * 2;
  const std::size_t meta_at = (svh_at + n * ref.src_out * 2 + 255) / 256 * 256;
  std::vector<std::uint64_t> meta(3 * n);
  for (std::size_t e = 0; e < n; ++e) {
    meta[e] = e * src;
    meta[n + e] = suh_at + e * ref.src_in * 2;
    meta[2 * n + e] = svh_at + e * ref.src_out * 2;
  }
  const std::size_t blocks_at = meta_at + meta.size() * 8;
  const std::size_t need =
      blocks_at + (ref.in_blocks.size() + ref.out_blocks.size()) * 4 + 64;
  if (need > staging_bytes_) {
    ReleaseStaging();
    if (hipMalloc(&staging_, need) != hipSuccess) {
      staging_ = nullptr;
      return fail("EXL3 staging allocation failed for " + std::string(t.name));
    }
    staging_bytes_ = need;
  }
  auto* base = static_cast<std::uint8_t*>(staging_);
  for (std::size_t e = 0; e < n; ++e) {
    const auto& part = ref.parts[e];
    if (!up.CopyFromRegion(part.file, part.file_offset, src, base + meta[e]) ||
        !up.CopyFromRegion(part.file, part.suh_offset, ref.src_in * 2,
                           base + meta[n + e]) ||
        !up.CopyFromRegion(part.file, part.svh_offset, ref.src_out * 2,
                           base + meta[2 * n + e]))
      return fail("upload failed for " + std::string(t.name));
  }
  if (!up.Finish())
    return fail("upload failed for " + std::string(t.name));
  std::vector<std::uint32_t> maps(ref.in_blocks);
  maps.insert(maps.end(), ref.out_blocks.begin(), ref.out_blocks.end());
  if (hipMemcpy(base + meta_at, meta.data(), meta.size() * 8,
                hipMemcpyHostToDevice) != hipSuccess ||
      (!maps.empty() &&
       hipMemcpy(base + blocks_at, maps.data(), maps.size() * 4,
                 hipMemcpyHostToDevice) != hipSuccess))
    return fail("EXL3 staging metadata upload failed");
  const auto* offsets = reinterpret_cast<const std::uint64_t*>(base + meta_at);
  const auto* in_map =
      ref.in_blocks.empty()
          ? nullptr
          : reinterpret_cast<const std::uint32_t*>(base + blocks_at);
  const auto* out_map =
      ref.out_blocks.empty()
          ? nullptr
          : reinterpret_cast<const std::uint32_t*>(base + blocks_at) +
                ref.in_blocks.size();
  // Every EXL3 linear has a one-row decode GEMV, which streams its tiles in
  // front order for its split. Linears launched beside the routed expert
  // stacks (the shared expert) take the stacks' split: their rows need not
  // fill the GPU alone (4.05 decode 35.3 -> 35.8 tok/s).
  const int front = rocm::Exl3DecodeSplits(static_cast<std::uint32_t>(in),
                                           static_cast<std::uint32_t>(out),
                                           n > 1 || hint.beside_routed_experts);
  rocm::Exl3Repack(base, offsets, static_cast<std::uint32_t*>(tiles),
                   ref.src_in, ref.src_out, static_cast<std::uint32_t>(in),
                   static_cast<std::uint32_t>(out), ref.k,
                   static_cast<std::uint32_t>(n), ref.DeviceBytes() / 4, in_map,
                   out_map, front, nullptr);
  rocm::Exl3GatherScales(base, offsets + n, static_cast<std::uint32_t>(n),
                         ref.src_in, in_map, static_cast<std::uint32_t>(in),
                         static_cast<__half*>(suh), nullptr);
  rocm::Exl3GatherScales(base, offsets + 2 * n, static_cast<std::uint32_t>(n),
                         ref.src_out, out_map, static_cast<std::uint32_t>(out),
                         static_cast<__half*>(svh), nullptr);
  // The staging buffer is reused by the next linear.
  if (hipDeviceSynchronize() != hipSuccess)
    return fail("EXL3 repack failed for " + std::string(t.name));
  const DeviceFormat& format = formats_.emplace_back(
      DeviceFormat{static_cast<const __half*>(suh),
                   static_cast<const __half*>(svh), ref.k, front});
  const fn::Tensor tensor(
      tiles, core::GgmlType::kI32, static_cast<std::uint32_t>(in),
      static_cast<std::uint32_t>(out), static_cast<std::uint32_t>(n), &format);
  if (draft_head) {
    // Draft tables copy head rows, which the trellis format cannot address:
    // the head drafts from the Q8_0 matrix it equals.
    if (hipMemset(static_cast<std::uint8_t*>(q8) + q8_bytes, 0, kTailMargin) !=
            hipSuccess ||
        !rocm::Exl3DequantizeQ8(WeightsOf(tensor), 0, q8, nullptr) ||
        hipDeviceSynchronize() != hipSuccess) {
      up.Fail("Q8_0 draft head expansion failed");
      return {};
    }
    draft_head_ =
        fn::Tensor(q8, core::GgmlType::kQ8_0, tensor.cols(), tensor.rows());
  }
  return tensor;
}

bool Exl3WeightSource::DraftHead(const fn::Tensor& output, fn::Tensor* head,
                                 std::string* error) {
  *head = {};
  if (!IsExl3(output))
    return true;
  if (draft_head_.empty() || draft_head_.rows() != output.rows() ||
      draft_head_.cols() != output.cols()) {
    if (error != nullptr)
      *error = "the EXL3 head was uploaded without its Q8_0 draft copy";
    return false;
  }
  *head = draft_head_;
  return true;
}

}  // namespace

std::unique_ptr<::gufo::ext::qwen::flash_next::WeightSource>
CreateExl3WeightSource() {
  return std::make_unique<Exl3WeightSource>();
}

}  // namespace gufo::ext::qwen38_flash_next_exl3
