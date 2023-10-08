#ifndef _UTIL_MATRIX_UTIL_H_
#define _UTIL_MATRIX_UTIL_H_

#include <string>
#include <tuple>
#include <vector>

#include "util/to_string.h"

namespace sungcho {

template <typename Elem>
std::string PrintVector(const Elem* elems, int32_t size, int32_t print_limit) {
  std::string ret = "[";
  for (int32_t i = 0; i < size; ++i) {
    if (i > 0) {
      ret += " ";
    }
    if (i >= print_limit) {
      ret += "...";
      break;
    }
    ret += ToString(elems[i]);
  }
  ret += "]";
  return ret;
}

template <typename Elem>
std::string PrintVector(const std::vector<Elem>& elems, int32_t print_limit) {
  return PrintVector(elems.data(), elems.size(), print_limit);
}

template <typename Elem>
std::string PrintMatrix(const Elem* elems, std::tuple<int32_t, int32_t> dims,
                        int32_t print_limit) {
  std::string ret = "[";
  const int32_t num_rows = std::get<0>(dims);
  const int32_t num_cols = std::get<1>(dims);
  for (int32_t row = 0; row < num_rows; ++row) {
    if (row >= print_limit) {
      ret += "...\n";
      break;
    }
    ret += PrintMatrixRow(&elems[num_cols * row], num_cols, print_limit);
    ret += "\n";
  }
  ret += "]";
  return ret;
}

}  // namespace sungcho

#endif  // _UTIL_MATRIX_UTIL_H_
