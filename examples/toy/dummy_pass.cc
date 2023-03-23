#include "examples/toy/dummy_pass.h"

#include <cassert>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "toy/Dialect.h"

namespace mlir::toy {

namespace {

constexpr char kMainFuncName[] = "main";

class DummyPass : public PassWrapper<DummyPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DummyPass)

  void runOnOperation() override {
    using TypeVector = llvm::SmallVector<Type, 4>;

    OpBuilder builder(&getContext());
    llvm::SmallVector<long int, 4> tensor_shape = {32, 64, 2};
    // Toy dialect requires the element type to be F64.
    Type element_type = builder.getF64Type();
    RankedTensorType tensor_type =
        RankedTensorType::get(tensor_shape, element_type);
    TypeVector input_types = {tensor_type};
    // Toy dialect requires the number of return values to be 1.
    TypeVector output_types = {tensor_type};
    FunctionType func_type =
        FunctionType::get(&getContext(), input_types, output_types);
    Location unknown_loc = builder.getUnknownLoc();

    ModuleOp module = getOperation();
    builder.setInsertionPointToEnd(module.getBody());

    // Build toy::FuncOp.
    auto func_op =
        builder.create<toy::FuncOp>(unknown_loc, kMainFuncName, func_type);
    Block* entry_block = &func_op.front();
    builder.setInsertionPointToStart(entry_block);
    // Build toy::ReturnOp.
    builder.create<toy::ReturnOp>(unknown_loc, entry_block->getArguments());

    if (failed(verify(module))) {
      // TODO(scho):
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateDummyPass() {
  return std::make_unique<DummyPass>();
}

}  // namespace mlir::toy

// namespace mlir::toy
