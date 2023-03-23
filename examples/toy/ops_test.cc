#include "absl/log/log.h"
#include "examples/toy/dummy_pass.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "toy/Dialect.h"

namespace mlir::toy {
namespace {

class ToyTest : public ::testing::Test {
 protected:
  const std::string func_name_ = "main";
  std::unique_ptr<MLIRContext> context_own_ = [] {
    auto context = std::make_unique<MLIRContext>();
    context->loadDialect<ToyDialect, func::FuncDialect>();
    return context;
  }();
  // Raw pointer just for convenience.
  MLIRContext* context_ = context_own_.get();
  OpBuilder builder_{context_};
};

TEST_F(ToyTest, Basic) {
  using TypeVector = llvm::SmallVector<Type, 4>;

  llvm::SmallVector<long int, 4> tensor_shape = {32, 64, 2};
  // Toy dialect requires the element type to be F64.
  Type element_type = builder_.getF64Type();
  RankedTensorType tensor_type =
      RankedTensorType::get(tensor_shape, element_type);
  TypeVector input_types = {tensor_type};
  // Toy dialect requires the number of return values to be 1.
  TypeVector output_types = {tensor_type};
  FunctionType func_type =
      FunctionType::get(context_, input_types, output_types);
  Location unknown_loc = builder_.getUnknownLoc();

  // Build ModuleOp.
  OwningOpRef<ModuleOp> module = builder_.create<ModuleOp>(unknown_loc);
  builder_.setInsertionPointToEnd(module->getBody());
  // Build toy::FuncOp.
  auto func_op =
      builder_.create<toy::FuncOp>(unknown_loc, func_name_, func_type);
  Block* entry_block = &func_op.front();
  builder_.setInsertionPointToStart(entry_block);
  // Build toy::ReturnOp.
  builder_.create<toy::ReturnOp>(unknown_loc, entry_block->getArguments());

  LOG(INFO) << debugString(*module);
  EXPECT_FALSE(failed(verify(*module))) << "module verification error";
}

class DummyPassTest : public ToyTest {
 protected:
  std::unique_ptr<PassManager> pm_ = [this] {
    auto pm = std::make_unique<PassManager>(context_);
    pm->addPass(CreateDummyPass());
    return pm;
  }();
};

TEST_F(DummyPassTest, Basic) {
  Location unknown_loc = builder_.getUnknownLoc();
  // Build ModuleOp.
  OwningOpRef<ModuleOp> module = builder_.create<ModuleOp>(unknown_loc);

  LOG(INFO) << "before: " << debugString(*module);
  pm_->run(module->getOperation());
  LOG(INFO) << "after: " << debugString(*module);
}

}  // namespace
}  // namespace mlir::toy
