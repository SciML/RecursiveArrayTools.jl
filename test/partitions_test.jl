using RecursiveArrayTools, Base.Test

A = (rand(5),rand(5))
p = ArrayPartition(A)

@test (p[1].x[1],p[1].x[2]) == (p.x[1][1],p.x[2][1])
p2 = similar(p)
p2[1] = 1
@test p2.x[1] != p.x[1]

C = rand(10)
p3 = similar(p,indices(C))
@test length(p3.x[1]) == length(p3.x[2]) == 10
@test length(p.x) == length(p2.x) == length(p3.x) == 2
