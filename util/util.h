#ifndef _UTIL_UTIL_H_
#define _UTIL_UTIL_H_

#include <cstdint>
#include <tuple>

namespace sungcho {

namespace util_internal {

template <int32_t... Indices>
struct Seq {};

template <int32_t n, int32_t... indices>
struct GenSeq : GenSeq<n - 1, n - 1, indices...> {};

template <int32_t... Indices>
struct GenSeq<0, Indices...> : Seq<Indices...> {};

template <typename T, typename Func, int32_t... Indices>
void ForEach(T&& t, Func func, Seq<Indices...>) {
  auto _ = {(func(std::get<Indices>(t)), 0)...};
}

}  // namespace util_internal

template <typename... Ts, typename Func>
void ForEachInTuple(const std::tuple<Ts...>& t, Func func) {
  util_internal::ForEach(t, func, util_internal::GenSeq<sizeof...(Ts)>());
}

}  // namespace sungcho

#endif  // _UTIL_UTIL_H_

