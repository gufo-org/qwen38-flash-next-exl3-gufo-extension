#include "flash_next/stages.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include "exl3/rocm/exl3.hpp"
#include "flash_next/device_format.hpp"

namespace gufo::ext::qwen38_flash_next_exl3 {
namespace {

namespace fn = ::gufo::ext::qwen::flash_next;
using fn::LinearInput;
using fn::LinearJob;
using fn::MoeRouting;
using fn::Phase;
using fn::StageContext;
using fn::Tensor;
using fn::TensorId;
using rocm::Exl3Weights;

constexpr std::size_t kAlign = 4096;

std::size_t AlignUp(std::size_t v) {
  return (v + kAlign - 1) / kAlign * kAlign;
}

bool Fail(const StageContext& ctx, const char* message) {
  if (ctx.error != nullptr)
    *ctx.error = message;
  return false;
}

/// Bytes of a layer's Q8_0 prompt expansion, as Exl3Stages::Expand lays it
/// out; 0 when the layer has no EXL3 dense projection, SIZE_MAX when its
/// projections cannot be stacked.
class Expansion {
public:
  Expansion(std::uint8_t* base, hipStream_t stream)
      : base_(base), stream_(stream) {}

  /// The parts stacked by rows into one Q8_0 matrix; each matrix starts
  /// aligned past the previous one's read tail.
  Tensor Stack(std::initializer_list<Tensor> parts) {
    if (!ok_)
      return {};
    std::uint32_t cols = parts.begin()->cols();
    std::uint32_t rows = 0;
    std::uint8_t* start = base_ != nullptr ? base_ + offset_ : nullptr;
    for (const Tensor& t : parts) {
      if (!IsExl3(t) || t.cols() != cols || t.experts() != 1) {
        ok_ = false;
        return {};
      }
      const Exl3Weights w = WeightsOf(t);
      if (base_ != nullptr &&
          !rocm::Exl3DequantizeQ8(w, 0, base_ + offset_, stream_)) {
        ok_ = false;
        return {};
      }
      offset_ += rocm::Exl3Q8Bytes(w);
      rows += t.rows();
    }
    offset_ = (offset_ + 4096 + 255) / 256 * 256;
    return Tensor(start, core::GgmlType::kQ8_0, cols, rows);
  }

  [[nodiscard]] bool ok() const noexcept { return ok_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return ok_ ? offset_ : SIZE_MAX;
  }

private:
  std::uint8_t* base_;
  hipStream_t stream_;
  std::size_t offset_{0};
  bool ok_{true};
};

/// The layer in the official GGUF layout: stacked Q8_0 projections of one
/// input and Q8_0 outputs, so prompt passes take its fused projections.
/// With `edit`, the expansion is written (on `stream`) and set on it.
std::size_t ExpandLayer(const fn::Layer& l, std::uint8_t* base,
                        hipStream_t stream, fn::LayerEdit* edit) {
  if (!(IsExl3(l.tensor(TensorId::kSsmQkv)) ||
        IsExl3(l.tensor(TensorId::kAttnQ))))
    return 0;
  Expansion x(base, stream);
  const auto set = [&](TensorId id, const Tensor& t) {
    if (edit != nullptr)
      edit->Set(id, t);
  };
  const auto one = [&](TensorId id) { set(id, x.Stack({l.tensor(id)})); };
  if (l.linear()) {
    set(TensorId::kSsmIn,
        x.Stack({l.tensor(TensorId::kSsmQkv), l.tensor(TensorId::kSsmGate)}));
    set(TensorId::kSsmQkv, {});
    set(TensorId::kSsmGate, {});
    one(TensorId::kSsmOut);
  } else {
    set(TensorId::kAttnQkv,
        x.Stack({l.tensor(TensorId::kAttnQ), l.tensor(TensorId::kAttnK),
                 l.tensor(TensorId::kAttnV)}));
    set(TensorId::kAttnQ, {});
    set(TensorId::kAttnK, {});
    set(TensorId::kAttnV, {});
    one(TensorId::kAttnOut);
    one(TensorId::kIndexerQ);
    one(TensorId::kIndexerK);
  }
  one(TensorId::kShexpGate);
  one(TensorId::kShexpUp);
  one(TensorId::kShexpDown);
  return x.bytes();
}

std::size_t MaxIn(const fn::Limits& limits) {
  std::size_t max_in = 0;
  const auto see = [&](const Tensor& t) {
    if (IsExl3(t))
      max_in = std::max<std::size_t>(max_in, t.cols());
  };
  const auto layer = [&](const fn::Layer& l) {
    if (!l.valid())
      return;
    for (auto id = static_cast<std::uint8_t>(TensorId::kSsmIn);
         id <= static_cast<std::uint8_t>(TensorId::kNextnFcHidden); ++id)
      see(l.tensor(static_cast<TensorId>(id)));
  };
  for (const fn::Layer& l : limits.layers)
    layer(l);
  layer(limits.mtp);
  see(limits.output);
  return max_in;
}

class Exl3Stages final : public fn::Stages {
public:
  [[nodiscard]] std::uint32_t Hooks() const override {
    return fn::kPromptLayers;
  }

  [[nodiscard]] std::size_t ScratchBytes(
      const fn::Limits& limits) const override {
    Layout layout;
    return Plan(limits, &layout) ? layout.total : 0;
  }

  bool Attach(const fn::Attachment& a, std::string* error) override {
    Layout layout;
    if (!Plan(a.limits, &layout)) {
      if (layout.total == SIZE_MAX && error != nullptr)
        *error = "EXL3 prompt layer expansion size";
      return layout.total != SIZE_MAX;
    }
    if (a.scratch == nullptr || a.scratch_bytes < layout.total) {
      if (error != nullptr)
        *error = "EXL3 stage scratch is missing";
      return false;
    }
    auto* base = static_cast<std::uint8_t*>(a.scratch);
    // The Q8_0 buffers' read tails and the split counters start zeroed.
    if (hipMemset(base, 0, layout.total) != hipSuccess ||
        hipDeviceSynchronize() != hipSuccess) {
      if (error != nullptr)
        *error = "EXL3 stage scratch initialization failed";
      return false;
    }
    scratch_ = {
        reinterpret_cast<__half*>(base + layout.xh),     layout.xh_elems,
        reinterpret_cast<float*>(base + layout.partial), kPartialElems,
        reinterpret_cast<unsigned*>(base + layout.done), kDoneElems};
    q8_ = base + layout.q8;
    layers_q8_ = base + layout.layers;
    layer_offset_ = std::move(layout.layer_offset);
    layer_ready_.assign(a.limits.layers.size(), 0);
    layers_ = a.limits.layers;
    return true;
  }

  bool Linear(const StageContext& ctx, const Tensor& w, const LinearInput& in,
              float* out) override {
    if (!IsExl3(w))
      return ctx.defaults.Linear(ctx, w, in, out);
    const Exl3Weights weights = WeightsOf(w);
    // Prompt passes take the matrix cores at every width, so their
    // arithmetic does not depend on how a prompt is chunked; decode and
    // verification keep the vector route. A dense matrix that fits the Q8_0
    // scratch is expanded per pass to its equivalent Q8_0 matrix and takes
    // the int8 route of Q8_0 weights; the output head and routed experts run
    // the trellis GEMM.
    const bool prompt = ctx.phase == Phase::kPrompt;
    const bool dense = in.ids == nullptr && in.slots == 1;
    if (prompt && dense && w.experts() == 1 &&
        rocm::Exl3Q8Bytes(weights) <= kQ8Bytes) {
      if (!rocm::Exl3DequantizeQ8(weights, 0, q8_, ctx.stream))
        return Fail(ctx, "EXL3 Q8_0 expansion failed");
      return ctx.defaults.Linear(
          ctx, Tensor(q8_, core::GgmlType::kQ8_0, w.cols(), w.rows()), in, out);
    }
    const bool ok =
        prompt && dense
            ? rocm::Exl3LinearGemm(weights, in.x, w.cols(), 1, nullptr, in.rows,
                                   out, w.rows(), in.rows, {}, scratch_,
                                   ctx.stream)
            : rocm::Exl3Linear(weights, in.x, w.cols(), in.slots, in.ids, out,
                               w.rows(), in.rows * in.slots, scratch_,
                               ctx.stream);
    return ok || Fail(ctx, "EXL3 projection failed");
  }

  bool LinearGroup(const StageContext& ctx, std::span<const LinearJob> jobs,
                   const LinearInput& in) override {
    if (in.ids != nullptr && jobs.size() == 2 && IsExl3(jobs[0].w) &&
        IsExl3(jobs[1].w)) {
      // Routed gate and up of every (token, slot) row in one launch.
      const Tensor& a = jobs[0].w;
      if (!rocm::Exl3LinearPair(WeightsOf(a), WeightsOf(jobs[1].w), in.x,
                                a.cols(), in.slots, in.ids, jobs[0].out,
                                jobs[1].out, a.rows(), in.rows * in.slots,
                                scratch_, ctx.stream))
        return Fail(ctx, "EXL3 expert pair projection failed");
      return true;
    }
    // Decode and verification (up to kExl3GemvRows rows): the linears of
    // one input in one launch.
    if (ctx.phase == Phase::kDecode && in.ids == nullptr && in.slots == 1 &&
        in.rows != 0 && in.rows <= rocm::kExl3GemvRows &&
        jobs.size() <= static_cast<std::size_t>(rocm::kExl3GroupJobs)) {
      rocm::Exl3Job group[rocm::kExl3GroupJobs];
      int n = 0;
      bool all = true;
      for (const LinearJob& job : jobs) {
        if (!IsExl3(job.w) || job.w.experts() != 1) {
          all = false;
          break;
        }
        group[n++] = {WeightsOf(job.w), in.x,    job.w.cols(), 1,
                      nullptr,          job.out, job.w.rows(), in.rows,
                      nullptr};
      }
      if (all && rocm::Exl3LinearGroup(group, n, scratch_, ctx.stream))
        return true;
    }
    for (const LinearJob& job : jobs) {
      if (!Linear(ctx, job.w, in, job.out))
        return false;
    }
    return true;
  }

  [[nodiscard]] bool MoeDefersEpilogue(const StageContext& ctx) override {
    // Prompt passes leave the routed experts' F16 rows to the combine.
    if (ctx.phase == Phase::kPrompt &&
        IsExl3(ctx.layer.tensor(TensorId::kFfnGateExps)))
      return true;
    return ctx.defaults.MoeDefersEpilogue(ctx);
  }

  bool MoeFfn(const StageContext& ctx, const MoeRouting& routing,
              const float* x, float* out) override {
    const fn::Layer& l = ctx.layer;
    if (!IsExl3(l.tensor(TensorId::kFfnGateExps)))
      return ctx.defaults.MoeFfn(ctx, routing, x, out);
    if (ctx.phase == Phase::kPrompt)
      return PromptMoe(ctx, routing, x, out);
    if (!routing.last_only && DecodeMoe(ctx, routing, x, out))
      return true;
    return ctx.defaults.MoeFfn(ctx, routing, x, out);
  }

  bool PromptLayer(const StageContext& ctx, fn::LayerEdit& edit,
                   bool* use) override {
    *use = false;
    const std::uint32_t il = ctx.layer.index();
    if (ctx.phase != Phase::kPrompt || layers_q8_ == nullptr ||
        il >= layer_ready_.size() || layer_offset_[il + 1] == layer_offset_[il])
      return true;
    if (layer_ready_[il] == 0) {
      // A layer is expanded on its first prompt pass and kept: the weights
      // never change. The edit reads the uploaded layer.
      const std::size_t bytes = ExpandLayer(
          layers_[il], layers_q8_ + layer_offset_[il], ctx.stream, &edit);
      if (bytes != layer_offset_[il + 1] - layer_offset_[il])
        return Fail(ctx, "EXL3 prompt layer expansion failed");
      layer_ready_[il] = 1;
    }
    *use = true;
    return true;
  }

private:
  /// Transformed inputs and split-K partials of the EXL3 linears; larger
  /// calls run in row chunks that fit. Routed prompt experts transform
  /// their rows inside the GEMM and need none.
  static constexpr std::size_t kMinXhElems = std::size_t{16} << 20;
  static constexpr std::size_t kPartialElems = std::size_t{32} << 20;
  static constexpr std::size_t kDoneElems = std::size_t{1} << 16;
  /// The largest dense trunk projection (q|gate) in Q8_0; the quantized
  /// tier over-reads 4096 bytes past it.
  static constexpr std::size_t kQ8Bytes = std::size_t{48} << 20;

  struct Layout {
    std::size_t xh{0};
    std::size_t xh_elems{0};
    std::size_t partial{0};
    std::size_t done{0};
    std::size_t q8{0};
    std::size_t layers{0};
    std::vector<std::size_t> layer_offset;  ///< [layers + 1], from `layers`
    std::size_t total{0};
  };

  /// False with total 0 when no tensor is EXL3, SIZE_MAX on failure.
  static bool Plan(const fn::Limits& limits, Layout* out) {
    const std::size_t max_in = MaxIn(limits);
    if (max_in == 0)
      return false;
    Layout& p = *out;
    p.xh_elems = std::max(kMinXhElems, std::size_t{limits.max_batch} * max_in);
    std::size_t at = 0;
    const auto take = [&](std::size_t bytes) {
      const std::size_t start = at;
      at = AlignUp(at + bytes);
      return start;
    };
    p.xh = take(p.xh_elems * sizeof(__half));
    p.partial = take(kPartialElems * sizeof(float));
    p.done = take(kDoneElems * sizeof(unsigned));
    p.q8 = take(kQ8Bytes + 4096);
    // Every layer's dense projections in Q8_0, each with its read tail,
    // filled by the layer's first prompt pass.
    p.layer_offset.assign(limits.layers.size() + 1, 0);
    for (std::size_t il = 0; il < limits.layers.size(); ++il) {
      const std::size_t bytes =
          ExpandLayer(limits.layers[il], nullptr, nullptr, nullptr);
      if (bytes == SIZE_MAX) {
        p.total = SIZE_MAX;
        return false;
      }
      p.layer_offset[il + 1] = p.layer_offset[il] + bytes;
    }
    p.layers = take(p.layer_offset.back());
    p.total = at;
    return true;
  }

  /// Decode and verification of an all-EXL3 MoE after routing: two grouped
  /// launches and the epilogue. False when the pass does not take that
  /// route.
  bool DecodeMoe(const StageContext& ctx, const MoeRouting& routing,
                 const float* x, float* out) {
    const fn::Layer& l = ctx.layer;
    const std::uint32_t n_tokens = ctx.n_tokens;
    const Tensor gate = l.tensor(TensorId::kFfnGateExps);
    const Tensor up = l.tensor(TensorId::kFfnUpExps);
    const Tensor down = l.tensor(TensorId::kFfnDownExps);
    const Tensor sh_gate = l.tensor(TensorId::kShexpGate);
    const Tensor sh_up = l.tensor(TensorId::kShexpUp);
    const Tensor sh_down = l.tensor(TensorId::kShexpDown);
    if (n_tokens == 0 || n_tokens > rocm::kExl3GemvRows || !IsExl3(up) ||
        !IsExl3(down) || !IsExl3(sh_gate) || !IsExl3(sh_up) || !IsExl3(sh_down))
      return false;
    const std::uint32_t hidden = gate.cols();
    const std::uint32_t ff = gate.rows();
    const std::uint32_t shared_ff = sh_gate.rows();
    const std::uint32_t used = routing.used;
    const std::uint32_t slots = n_tokens * used;
    const auto& s = routing.scratch;
    // Gate and up of the routed (token, slot) rows and of the shared expert,
    // then both down projections with Swiglu formed while their inputs are
    // staged: the arithmetic of GatedExperts, GatedDense, Experts and Dense.
    const rocm::Exl3Job jobs[6] = {
        {WeightsOf(gate), x, hidden, used, routing.ids, s.gate, ff, slots,
         nullptr},
        {WeightsOf(up), x, hidden, used, routing.ids, s.up, ff, slots, nullptr},
        {WeightsOf(sh_gate), x, hidden, 1, nullptr, s.shared_up, shared_ff,
         n_tokens, nullptr},
        {WeightsOf(sh_up), x, hidden, 1, nullptr, s.shared_gate, shared_ff,
         n_tokens, nullptr},
        {WeightsOf(down), s.up, ff, 1, routing.ids, s.down, hidden, slots,
         s.gate},
        {WeightsOf(sh_down), s.shared_gate, shared_ff, 1, nullptr, s.shared_out,
         hidden, n_tokens, s.shared_up}};
    if (!rocm::Exl3LinearGroup(jobs, 4, scratch_, ctx.stream) ||
        !rocm::Exl3LinearGroup(jobs + 4, 2, scratch_, ctx.stream))
      return false;
    ctx.defaults.MoeEpilogue(ctx, routing, out, false);
    return true;
  }

  /// Prompt passes: assignments compacted by expert into 16-row padded
  /// buckets, then matrix-core products per (expert, 128-row tile): gate
  /// and up transform their F16 token rows, down reads up's. The shared
  /// expert (Q8_0 after PromptLayer) is forked after gate and up when it
  /// overlaps: their 640-column blocks each take a whole WGP and share it
  /// poorly, while down's small blocks interleave with the shared expert.
  bool PromptMoe(const StageContext& ctx, const MoeRouting& routing,
                 const float* x, float* out) {
    const fn::Layer& l = ctx.layer;
    const std::uint32_t n_tokens = ctx.n_tokens;
    const Tensor gate = l.tensor(TensorId::kFfnGateExps);
    const Tensor up = l.tensor(TensorId::kFfnUpExps);
    const Tensor down = l.tensor(TensorId::kFfnDownExps);
    if (!IsExl3(up) || !IsExl3(down))
      return Fail(ctx, "EXL3 routed experts must share one format");
    const bool fork = routing.overlap && !routing.last_only;
    if (!fork && !ctx.defaults.SharedExpert(ctx, routing, x, false))
      return false;
    if (!ctx.defaults.WaitCounts(ctx))
      return false;
    const std::uint32_t hidden = gate.cols();
    const std::uint32_t ff = gate.rows();
    const std::uint32_t used = routing.used;
    const std::uint32_t slots = n_tokens * used;
    const auto& s = routing.scratch;
    // Both tile maps are one upload.
    std::int32_t* tiles = routing.host_tiles;
    std::size_t n_tiles = 0;
    std::size_t n_down_tiles = 0;
    for (std::uint32_t pass = 0; pass < 2; ++pass) {
      const std::uint32_t rows =
          pass == 0 ? rocm::kExl3GemmRows : rocm::kExl3TransformedGemmRows;
      std::size_t& count = pass == 0 ? n_tiles : n_down_tiles;
      for (std::uint32_t e = 0; e < routing.num_experts; ++e) {
        for (std::uint32_t j = 0; j * rows < routing.host_counts[e]; ++j) {
          if (n_tiles + n_down_tiles >= routing.host_tiles_capacity)
            return Fail(ctx, "EXL3 routed tile map exceeds its staging");
          tiles[n_tiles + n_down_tiles] =
              static_cast<std::int32_t>(e | (j << 16));
          ++count;
        }
      }
    }
    if (n_tiles != 0 &&
        hipMemcpyAsync(s.routed_tiles, tiles,
                       (n_tiles + n_down_tiles) * sizeof(std::int32_t),
                       hipMemcpyHostToDevice, ctx.stream) != hipSuccess)
      return Fail(ctx, "routed tile map upload failed");
    ctx.defaults.CompactRoutes(ctx, routing);
    const auto* x_half =
        static_cast<const __half*>(ctx.defaults.HalfRows(ctx, x, hidden));
    const rocm::Exl3GemmRoute from_tokens{
        s.routed_tiles, static_cast<std::uint32_t>(n_tiles), s.routed_bounds,
        s.rows_token, s.rows_slot};
    const rocm::Exl3GemmRoute from_slots{
        s.routed_tiles + n_tiles, static_cast<std::uint32_t>(n_down_tiles),
        s.routed_bounds, s.rows_slot, s.rows_slot};
    // Gate and up read the F16 token rows and transform them in the GEMM
    // with the tile's expert suh; compacted positions write slots. Up
    // applies SwiGLU and down's input transform into F16 down inputs, and
    // down leaves F16 rows for the combine's MoE epilogue.
    auto* act = reinterpret_cast<__half*>(s.up);
    auto* down_half = reinterpret_cast<__half*>(s.down);
    rocm::Exl3GemmOut gate_out{};
    gate_out.x_half = x_half;
    gate_out.fused_input = true;
    rocm::Exl3GemmOut up_out{};
    up_out.x_half = x_half;
    up_out.y_half = act;
    up_out.gate = s.gate;
    up_out.fused_input = true;
    up_out.next_suh = WeightsOf(down).suh;
    rocm::Exl3GemmOut down_out{};
    down_out.x_half = act;
    down_out.y_half = down_half;
    down_out.x_transformed = true;
    if (!rocm::Exl3LinearGemm(WeightsOf(gate), x, hidden, used, routing.ids,
                              slots, s.gate, ff, 0, from_tokens, scratch_,
                              ctx.stream, gate_out) ||
        !rocm::Exl3LinearGemm(WeightsOf(up), x, hidden, used, routing.ids,
                              slots, nullptr, ff, 0, from_tokens, scratch_,
                              ctx.stream, up_out))
      return Fail(ctx, "routed EXL3 expert GEMM failed");
    if (fork && !ctx.defaults.SharedExpert(ctx, routing, x, true))
      return false;
    if (!rocm::Exl3LinearGemm(WeightsOf(down), nullptr, ff, 1, routing.ids,
                              slots, nullptr, hidden, 0, from_slots, scratch_,
                              ctx.stream, down_out))
      return Fail(ctx, "routed EXL3 expert GEMM failed");
    // The epilogue reads the shared expert's output.
    if (!ctx.defaults.JoinSide(ctx))
      return false;
    ctx.defaults.MoeEpilogue(ctx, routing, out, true);
    return true;
  }

  rocm::Exl3Scratch scratch_{};
  std::uint8_t* q8_{nullptr};
  std::uint8_t* layers_q8_{nullptr};
  std::vector<std::size_t> layer_offset_;
  std::vector<std::uint8_t> layer_ready_;
  std::span<const fn::Layer> layers_;
};

}  // namespace

std::unique_ptr<::gufo::ext::qwen::flash_next::Stages> CreateExl3Stages() {
  return std::make_unique<Exl3Stages>();
}

}  // namespace gufo::ext::qwen38_flash_next_exl3
