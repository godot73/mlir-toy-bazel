#ifndef CONV1D_H_NGJO1SFU
#define CONV1D_H_NGJO1SFU

#include <vector>

namespace sungcho {

int get_output_size(int input_size, int kernel_size, int stride);

std::vector<float> conv1d(const std::vector<float>& image,
                          const std::vector<float>& filter, int stride);

}  // namespace sungcho

#endif /* end of include guard: CONV1D_H_NGJO1SFU */

