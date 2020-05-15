#pragma once
#include <ATen/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <array>

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)

namespace at {
namespace native {
namespace vulkan {

struct Conv2DParams final {
  int64_t N; // batch size
  int64_t C; // channels
  int64_t H; // input height
  int64_t W; // input width
  int64_t OC; // output channels
  int64_t KH; // kernel height
  int64_t KW; // kernel width
  int64_t SY; // stride y (height)
  int64_t SX; // stride x (width)
  int64_t PY; // padding y (height)
  int64_t PX; // padding y (height)
  int64_t DY; // dilation y (height)
  int64_t DX; // dilation y (height)
  int64_t G; // groups
  int64_t OW; // output width
  int64_t OH; // output height
  int64_t OC_4;
  int64_t C_4;

  Conv2DParams() = delete;
  Conv2DParams(
      c10::IntArrayRef inputSizes,
      int64_t OC,
      int64_t KH,
      int64_t KW,
      int64_t SY,
      int64_t SX,
      int64_t PY,
      int64_t PX,
      int64_t DY,
      int64_t DX,
      int64_t G)
      : N(inputSizes[0]),
        C(inputSizes[1]),
        H(inputSizes[2]),
        W(inputSizes[3]),
        OC(OC),
        KH(KH),
        KW(KW),
        SY(SY),
        SX(SX),
        PY(PY),
        PX(PX),
        DY(DY),
        DX(DX),
        G(G) {
    OC_4 = UP_DIV(OC, 4);
    C_4 = UP_DIV(C, 4);
    const int64_t KWE = (KW - 1) * DX + 1;
    const int64_t KHE = (KH - 1) * DY + 1;
    OW = ((W - KWE + 2 * PX) / SX) + 1;
    OH = ((H - KHE + 2 * PY) / SY) + 1;
  }

  Conv2DParams(
      c10::IntArrayRef inputSizes,
      c10::IntArrayRef weightSizes,
      c10::IntArrayRef padding,
      c10::IntArrayRef stride,
      c10::IntArrayRef dilation,
      int64_t groups)
      : Conv2DParams(
            inputSizes,
            weightSizes[0],
            weightSizes[2],
            weightSizes[3],
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
            groups) {}

  std::vector<int64_t> output_sizes() {
    return {N, OC, OH, OW};
  }
};

struct ContextConv2D final {
  at::Tensor weight_prepacked_vulkan_;
  c10::optional<at::Tensor> bias_vulkan_;
  std::array<int64_t, 4> weight_size_;
  std::array<int64_t, 2> padding_;
  std::array<int64_t, 2> stride_;
  std::array<int64_t, 2> dilation_;
  int64_t groups_;

  ContextConv2D() = delete;

  ContextConv2D(
      at::Tensor&& weight_prepacked_vulkan,
      c10::optional<at::Tensor>&& bias_vulkan,
      std::array<int64_t, 4> weight_size,
      std::array<int64_t, 2> padding,
      std::array<int64_t, 2> stride,
      std::array<int64_t, 2> dilation,
      int64_t groups)
      : weight_prepacked_vulkan_(std::move(weight_prepacked_vulkan)),
        bias_vulkan_(std::move(bias_vulkan)),
        weight_size_(weight_size),
        padding_(padding),
        stride_(stride),
        dilation_(dilation),
        groups_(groups) {}

  ContextConv2D(ContextConv2D&&) = default;
  ContextConv2D& operator=(ContextConv2D&&) = default;

  ~ContextConv2D() {}

  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
};

} // namespace vulkan
} // namespace native
} // namespace at
