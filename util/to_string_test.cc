#include "to_string.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace sungcho {
namespace {

TEST(ToStringTest, String) {
  const std::string s = "hello";
  EXPECT_EQ(ToString(s), "\"hello\"");
}

TEST(ToStringTest, CstrLiteral) {
  EXPECT_EQ(ToString("hello"), "\"hello\"");
}

TEST(ToStringTest, VectorInt32) {
  const std::vector<int32_t> vs = {-2, -1, 0, 1, 2};
  EXPECT_EQ(ToString(vs), "{-2,-1,0,1,2}");
}

TEST(ToStringTest, VectorString) {
  const std::vector<std::string> vs = {"One", "Two", "Three"};
  EXPECT_EQ(ToString(vs), R"({"One","Two","Three"})");
}

TEST(ToStringTest, VectorCString) {
  const std::vector<const char*> vs = {"One", "Two", "Three"};
  EXPECT_EQ(ToString(vs), R"({"One","Two","Three"})");
}

TEST(ToStringTest, VectorFloat) {
  const std::vector<float> vs = {-2, -1, 0, 1, 2};
  EXPECT_EQ(ToString(vs), "{-2,-1,0,1,2}");
}

TEST(ToStringTest, VectorDouble) {
  const std::vector<double> vs = {-2, -1, 0, 1, 2};
  EXPECT_EQ(ToString(vs), "{-2,-1,0,1,2}");
}

TEST(ToStringTest, VectorInt16) {
  const std::vector<int16_t> vs = {-2, -1, 0, 1, 2};
  EXPECT_EQ(ToString(vs), "{-2,-1,0,1,2}");
}

TEST(ToStringTest, VectorInt8) {
  const std::vector<int8_t> vs = {-2, -1, 0, 1, 2};
  EXPECT_EQ(ToString(vs), "{-2,-1,0,1,2}");
}

TEST(ToStringTest, NestedVectorInt32) {
  const std::vector<std::vector<int32_t>> vs = {{-2, -1, 0, 1, 2}, {-1, 0, 1}};
  EXPECT_EQ(ToString(vs), "{{-2,-1,0,1,2},{-1,0,1}}");
}

TEST(ToStringTest, SetInt32) {
  const std::set<int32_t> vs = {-2, -1, 0, 1, 2};
  EXPECT_EQ(ToString(vs), "{-2,-1,0,1,2}");
}

TEST(ToStringTest, MapInt32String) {
  const std::map<int32_t, std::string> kvs = {
      {1, "One"}, {2, "Two"}, {3, "Three"}};
  EXPECT_EQ(ToString(kvs), R"({{1,"One"},{2,"Two"},{3,"Three"}})");
}

TEST(ToStringTest, PairIntString) {
  const std::pair<int32_t, std::string> p = {1, "One"};
  EXPECT_EQ(ToString(p), R"({1,"One"})");
}

TEST(ToStringTest, TupleIntStringDouble) {
  const std::tuple<int32_t, std::string, double> t = {2, "Two", 0.5};
  EXPECT_EQ(ToString(t), R"({2,"Two",0.5})");
}

}  // namespace
}  // namespace sungcho

