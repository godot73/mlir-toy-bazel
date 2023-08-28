func.func  @matmul_aie_offload(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %dest = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  // Create a dispatch that operates on `2` threads. Set the `workload` of
  // `flow.dispatch.region` to capture the values needed to specify the number
  // of threads to use in the `count` region.
  %0 = flow.dispatch.region[%c2] -> (tensor<?x?xf32>{%d0, %d1}) {
    %id = flow.dispatch.workgroup.id[0] : index
    %count = flow.dispatch.workgroup.count[0] : index

    // Each thread has to perform a slice of the computation.
    %tilesize = affine.apply affine_map<()[s0, s1] -> (s0 ceildiv s1)>()[%d0, %count]

    // Compute the offset and size of the slice
    %offset_0 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%id, %tilesize]
    %offset_1 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%id, %tilesize]
    %size_0 = affine.min affine_map<(d0)[s0, s1] -> (s1 - d0, s0)>(%offset_0)[%tilesize, %d0]
    %size_1 = affine.min affine_map<(d1)[s0, s1] -> (s1 - d1, s0)>(%offset_1)[%tilesize, %d1]

    // Extract slices of the inputs and outputs.
    %1 = tensor.extract_slice %arg0[%offset_0, %offset_1] [%size_0, %size_1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    %2 = tensor.extract_slice %arg1[%offset_0, %offset_1] [%size_0, %size_1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    %3 = tensor.extract_slice %dest[%offset_0, %offset_1] [%size_0, %size_1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

    // Invoke the ukernel.
    %4 = iree_codegen.ukernel.generic "aie_matmul_f32"
      ins(%1, %2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%3 : tensor<?x?xf32>)
      (%size_0, %size_1, %id : index, index, index) 
      // We can include some additional fields on the parameters struct as
      // needed. Here we request which processor is executing the call and
      // its data fields as defined by runtime/src/iree/schemas/cpu_data.h.
      fn_def_attrs {hal.import.fields = ["processor_id", "processor_data"]}
      // Set the operation to not incorporate any strides. The implementation
      // expects no stride arguments.
      strided_outer_dims(0) -> tensor<?x?xf32>

    // Insert the result back into the result at the right position.
    %5 = tensor.insert_slice %4 into %dest[%offset_0, %offset_1] [%size_0, %size_1] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
    flow.return %5 : tensor<?x?xf32>
  } count(%b0 : index) -> (index, index, index) {
    // Specify the number of threads to use. `%b0` represents
    // the values captured as workload (within `[` `]` in the `flow.dispatch.region` above)
    // Use that to derive the number of threads to use along `x`, `y` and `z`.
    flow.return %b0, %c1, %c1 : index, index, index
  }
  return %0 : tensor<?x?xf32>
}
