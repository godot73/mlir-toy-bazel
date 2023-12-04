TEMPL = """
func.func @matmul() -> tensor<{m}x{n}xf32>
{{
  %arg0 = util.unfoldable_constant dense<1.0> : tensor<{m}x{k}xf32>
  %arg1 = util.unfoldable_constant dense<0.4> : tensor<{k}x{n}xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<{m}x{n}xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>)
    outs(%1: tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
  return %2: tensor<{m}x{n}xf32>
}}
"""

if __name__ == '__main__':
    params = {'m': 32, 'k': 64, 'n': 128}
    print(TEMPL.format(**params))
