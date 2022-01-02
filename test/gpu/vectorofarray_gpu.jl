using RecursiveArrayTools, CUDA

x = zeros(5)
y = VectorOfArray([x,x,x])
y[:,:]

x = CUDA.zeros(5)
y = VectorOfArray([x,x,x])
y[:,:]
