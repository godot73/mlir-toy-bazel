module @matmul_4x2048x2048_toy_offload {
 //===--------------------------------------------------------------------===//
  // Imports
 //===--------------------------------------------------------------------===//
  // External function declarations for the methods implemented in the custom
  // module C++ file. Note that they are prefixed with the `custom.` module
  // name.

  func.func private @custom.toy_aie_matmul_f32(%lhs : tensor<4x2048xf32>, %rhs : tensor<2048x2048xf32>) -> tensor<4x2048xf32>

 //===--------------------------------------------------------------------===//
  // Sample methods
 //===--------------------------------------------------------------------===//
  // Note that there can be any number of publicly-exported methods; this
  // sample just has one to keep things simple.

  func.func @main() {
    %toy_activation = util.unfoldable_constant dense<1.0> : tensor<4x2048xf32>
    %toy_parameters = util.unfoldable_constant dense<1.0> : tensor<2048x2048xf32>
    %toy_result = call @custom.toy_aie_matmul_f32(%toy_activation, %toy_parameters) : (tensor<4x2048xf32>, tensor<2048x2048xf32>) -> tensor<4x2048xf32>

    return
   }
}
