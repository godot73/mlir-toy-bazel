func.func @matmul() -> tensor<32x128xf32>
{
  %arg0 = util.unfoldable_constant dense<1.0> : tensor<32x64xf32>
  %arg1 = util.unfoldable_constant dense<0.4> : tensor<64x128xf32>
  %arg2 = util.unfoldable_constant dense<0.3> : tensor<128x128xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<32x128xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x128xf32>) -> tensor<32x128xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x128xf32>)
    outs(%1: tensor<32x128xf32>) -> tensor<32x128xf32>

  %3 = tensor.empty() : tensor<32x128xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<32x128xf32>) -> tensor<32x128xf32>

  %5 = linalg.matmul ins(%2, %arg2 : tensor<32x128xf32>, tensor<128x128xf32>)
    outs(%4: tensor<32x128xf32>) -> tensor<32x128xf32>
  return %5: tensor<32x128xf32>
}


