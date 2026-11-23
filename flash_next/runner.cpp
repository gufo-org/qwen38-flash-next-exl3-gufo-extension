#include "flash_next/runner.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <gufo/ext/families/qwen/vision/encoder.hpp>
#include <gufo/ext/sha256.hpp>
#include <optional>
#include <string>
#include <utility>

#include "exl3/pack.hpp"
#include "flash_next/stages.hpp"
#include "flash_next/weight_source.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/rope_scaling.hpp"
#include "src/models/qwen38_flash_next/runner.hpp"

namespace gufo::ext::qwen38_flash_next_exl3 {

namespace flash_next = ::gufo::models::qwen38_flash_next;

std::shared_ptr<TextModelRunner> CreateExl3TextRunner(
    const TextRunnerOptions& options, std::string* error) {
  const auto fail = [&](std::string message) {
    if (error != nullptr)
      *error = std::move(message);
    return std::shared_ptr<TextModelRunner>();
  };
  if (!IsFlashNextPack(options.model_path))
    return fail("not a Qwen3.8-Flash-Next EXL3 checkpoint directory: " +
                options.model_path);
  const std::uint32_t native_context = PackContextLength(options.model_path);
  std::uint32_t max_context = options.max_context;
  if (max_context == 0) {
    max_context = native_context;
    if (max_context < 2)
      return fail(
          "checkpoint has no valid native context length; specify "
          "--context");
  }
  std::optional<flash_next::RopeScaling> rope_scaling;
  if (!flash_next::ResolveRopeScaling(max_context, native_context,
                                      options.enable_yarn, &rope_scaling,
                                      error))
    return nullptr;
  const bool use_mtp =
      options.speculative.backend == TextSpeculativeBackend::kMtp;
  if (options.speculative.backend != TextSpeculativeBackend::kDisabled &&
      !use_mtp)
    return fail(
        "Qwen3.8-Flash-Next HTTP models support only MTP speculative "
        "decoding");
  std::string load_error;
  // The checkpoint's own MTP block serves its drafts: the draft path names
  // the checkpoint.
  auto model = flash_next::Model::Load(
      options.model_path,
      flash_next::ModelOptions{
          .max_context = max_context,
          .mtp_model_path = use_mtp ? options.model_path : std::string{},
          .vision_model_path = options.vision_model_path,
          .decode_concurrency =
              flash_next::DecodeConcurrencyForSessions(options.session_count),
          .rope_scaling = rope_scaling,
      },
      CreateExl3Stages(), CreateExl3WeightSource(), &load_error);
  if (!model)
    return fail("Failed to load Flash-Next model: " + load_error);
  try {
    ::gufo::models::qwen::vision::PrepareForServing(model->VisionEncoder());
  } catch (const std::exception& e) {
    return fail(e.what());
  }
  // Each eight rows of wide copy verification past the first reserve
  // rollback rows beside the sessions; narrow the width until the
  // configured sessions fit (as Flash-Next's GGUF runner does).
  if (std::size_t free_bytes = 0, total_bytes = 0;
      use_mtp && mem::MemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
    const std::size_t sessions =
        std::max<std::size_t>(options.session_count, 1);
    const auto mode = core::SessionMode::kSpeculative;
    const auto fit = [&](std::uint32_t rows) {
      model->LimitCopyRows(rows);
      const std::size_t deferred = model->DeferredScratchBytes();
      if (free_bytes <= deferred)
        return mem::SessionFit::Kind::kRefused;
      return mem::FitSessions(sessions, model->SessionBytes(mode, max_context),
                              model->DemandMappedBytes(mode, max_context),
                              free_bytes - deferred)
          .kind;
    };
    std::uint32_t rows = flash_next::rocm::Executor::kWideRows;
    while (rows > 8 && fit(rows) != mem::SessionFit::Kind::kFits)
      rows -= 8;
    if (fit(rows) != mem::SessionFit::Kind::kFits)
      for (rows = flash_next::rocm::Executor::kWideRows;
           rows > 8 && fit(rows) == mem::SessionFit::Kind::kRefused;)
        rows -= 8;
    model->LimitCopyRows(rows);
  }
  std::string fingerprint = options.disk_cache.model_artifact_fingerprint;
  if (fingerprint.empty())
    fingerprint = model->ArtifactIdentity();
  std::string draft = options.disk_cache.draft_model_artifact_fingerprint;
  if (use_mtp && draft.empty()) {
    // The draft block is part of the checkpoint: derive its identity.
    const std::string seed = model->ArtifactIdentity() + ":mtp";
    draft = crypto::Sha256Hex(
        {reinterpret_cast<const std::uint8_t*>(seed.data()), seed.size()});
  }
  return flash_next::CreateQwenFlashNextTextRunner(
      std::move(model), max_context, options.speculative, fingerprint, draft);
}

}  // namespace gufo::ext::qwen38_flash_next_exl3
