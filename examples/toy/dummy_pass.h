#ifndef EXAMPLES_TOY_DUMMY_PASS_H_
#define EXAMPLES_TOY_DUMMY_PASS_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir::toy {

std::unique_ptr<Pass> CreateDummyPass();

}  // namespace mlir::toy

#endif  // EXAMPLES_TOY_DUMMY_PASS_H_
