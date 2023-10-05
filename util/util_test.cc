#include "util/util.h"

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

namespace sungcho {
namespace {

TEST(UtilTest, ForEachTestBasic) {
  const std::tuple<int32_t, std::string, double> p = {2, "Two", .5};
  std::string s;
  ForEachInTuple(p, [&](const auto& v) { s += absl::StrCat(v); });
  EXPECT_EQ(s, "2Two0.5");
}

}  // namespace
}  // namespace sungcho
