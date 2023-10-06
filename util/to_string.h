#ifndef _UTIL_TO_STRING_H_
#define _UTIL_TO_STRING_H_

#include <map>
#include <set>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "util.h"

namespace sungcho {

namespace internal {
template <typename T>
std::string IterableToString(const T& vs);
}  // namespace internal

template <typename T>
std::string ToString(const T& t) {
  // Specialization hack without using concepts (available from C++20)
  if constexpr (std::is_arithmetic<T>::value) {
    return absl::StrCat(t);
  }
  const auto size = sizeof(t);
  const uint8_t* head = reinterpret_cast<const uint8_t*>(&t);
  std::string ret;
  for (int i = 0; i < size; ++i) {
    ret += absl::StrFormat("<%02X>", *(head + i));
  }
  return ret;
}

template <>
std::string ToString(const char* const& v) {
  return absl::StrCat("\"", v, "\"");
}

template <>
std::string ToString(const std::string& v) {
  return ToString(v.c_str());
}

template <typename T>
std::string ToString(const std::vector<T>& vs) {
  return internal::IterableToString(vs);
}

template <typename T>
std::string ToString(const std::set<T>& vs) {
  return internal::IterableToString(vs);
}

template <typename First, typename Second>
std::string ToString(const std::pair<First, Second>& p) {
  return absl::StrFormat("[%s,%s]", ToString(p.first), ToString(p.second));
}

template <typename Key, typename Value>
std::string ToString(const std::map<Key, Value>& kvs) {
  return internal::IterableToString(kvs);
}

template <typename... T>
std::string ToString(const std::tuple<T...>& t) {
  std::vector<std::string> ss;
  ForEachInTuple(t, [&](const auto& v) { ss.push_back(ToString(v)); });
  return absl::StrCat("[", absl::StrJoin(ss, ","), "]");
}

namespace internal {

template <typename T>
std::string IterableToString(const T& vs) {
  std::string ret = "[";
  bool first = true;
  for (const auto& v : vs) {
    if (!first) {
      ret += ",";
    }
    ret += ToString(v);
    first = false;
  }
  ret += "]";
  return ret;
}

}  // namespace internal

}  // namespace sungcho

#endif  // _UTIL_TO_STRING_H_
