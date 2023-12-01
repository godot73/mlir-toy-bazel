#include "examples/conv/conv1d.h"

namespace sungcho {

int get_output_size(int input_size, int kernel_size, int stride) {
  // TODO(scho):
  // wi - 1 = (wo - 1) * stride + kernel_size - 1
  // wi = (wo - 1) * stride + kernel_size
  // (wo - 1) * stride = wi - kernel_size
  // wo = (wi - kernel_size) / stride + 1
  return (input_size - kernel_size) / stride + 1;
}

std::vector<float> conv1d(const std::vector<float>& image,
                          const std::vector<float>& filter, int stride) {
  // TODO(scho):
  return {};
}

}  // namespace sungcho
