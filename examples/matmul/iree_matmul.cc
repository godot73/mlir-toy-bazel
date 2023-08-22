#include "examples/matmul/matmul.h"

#include <cstdlib>
#include <cstdio>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/vm/api.h"
#include "iree/vm/native_module_cc.h"
#include "iree/vm/dynamic/api.h"

// NOTE: this module is written in C++ using the native module wrapper and uses
// template magic to handle marshaling arguments. For a lot of uses this is a
// much friendlier way of exposing modules to the IREE VM and if performance and
// code size are not a concern is a fine route to take. Here we do it for
// brevity but all of the internal IREE modules are implemented in C.
//
//===----------------------------------------------------------------------===//
// !custom.matmul type
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

namespace {

using namespace iree;

// Per-context module state.
// This can contain "globals" and other arbitrary state.
//
// Thread-compatible; the runtime will not issue multiple calls at the same
// time using the same state. If the implementation uses external threads then
// it must synchronize itself.
class CustomModuleState final {
 public:
  explicit CustomModuleState(vm::ref<iree_hal_device_t> device,
		  	     iree_allocator_t host_allocator)
      : device_(std::move(device)), host_allocator_(host_allocator) {}
  ~CustomModuleState() = default;

  // Performs matmuls on a set of tensors
  StatusOr<vm::ref<iree_hal_buffer_view_t>> MatmulF32(
	vm::ref<iree_hal_buffer_view_t> buffer_view) {
    vm::ref<iree_hal_buffer_t> result_buffer;

