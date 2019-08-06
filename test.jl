using RecursiveArrayTools, BenchmarkTools

function Base.copyto!(A::ArrayPartition,src::ArrayPartition)
    @assert length(src) == length(A)
    if size.(A.x) == size.(src.x)
      A .= src
    else
      cnt = 0
      for i in eachindex(A.x)
        x = A.x[i]
        for k in eachindex(x)
          cnt += 1
          x[k] = src[cnt]
        end
      end
    end
    A
end

x = ArrayPartition(randn(1000,10),randn(1000),randn(3)); 
y =zero(x); 
@btime copyto!($y,$x);
@btime $y .= $x;
