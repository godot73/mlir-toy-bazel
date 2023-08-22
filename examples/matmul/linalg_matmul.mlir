func.func @matmul() -> tensor<4x2048xf32>
{
  %arg0 = util.unfoldable_constant dense<1.0> : tensor<4x2048xf32>
  %arg1 = util.unfoldable_constant dense<0.4> : tensor<2048x2048xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<4x2048xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x2048xf32>) -> tensor<4x2048xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<4x2048xf32>, tensor<2048x2048xf32>)
    outs(%1: tensor<4x2048xf32>) -> tensor<4x2048xf32>
  return %2: tensor<4x2048xf32>
}

