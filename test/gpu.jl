using RecursiveArrayTools, CuArrays
a = ArrayPartition(([1.0f0] |> cu,[2.0f0] |> cu,[3.0f0] |> cu))
b = ArrayPartition(([0.0f0] |> cu,[0.0f0] |> cu,[0.0f0] |> cu))
@. a + b
