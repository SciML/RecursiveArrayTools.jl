using RecursiveArrayTools
using RecursiveArrayToolsShorthandConstructors
using Test

@testset "Documented constructors and utilities" begin
    vector_of_array = VectorOfArray([[1, 2], [3, 4]])
    @test size(vector_of_array) == (2, 2)
    @test Array(vector_of_array) == [1 3; 2 4]
    @test vecarr_to_vectors(vector_of_array) == [[1, 3], [2, 4]]

    diffeq_array = DiffEqArray([[1.0], [2.0]], [0.0, 1.0])
    @test tuples(diffeq_array) == [(0.0, [1.0]), (1.0, [2.0])]

    partition = ArrayPartition([1, 2], [3.0, 4.0])
    @test collect(partition) == [1.0, 2.0, 3.0, 4.0]
    @test AP[[1, 2], [3.0, 4.0]] == partition
    @test VA[[1, 2], [3, 4]] == vector_of_array

    named_partition = NamedArrayPartition(
        position = [1.0, 2.0], velocity = [3.0, 4.0]
    )
    @test named_partition.position == [1.0, 2.0]
    @test vecvec_to_mat([[1, 2], [3, 4]]) == [1 2; 3 4]

    values = [[1, 2]]
    copyat_or_push!(values, 2, [3, 4])
    @test values == [[1, 2], [3, 4]]
    @test recursive_one([[2.0]]) == 1.0
end

@testset "Plot recipe developer interface" begin
    A = DiffEqArray([[1.0, 2.0], [3.0, 4.0]], [0.0, 1.0])

    @test DEFAULT_PLOT_FUNC(1, 2) == (1, 2)
    @test DEFAULT_PLOT_FUNC(1, 2, 3) == (1, 2, 3)
    @test plottable_indices([1, 2]) == 1:2
    @test plottable_indices(1) == 1
    @test plot_indices([1, 2]) == eachindex([1, 2])
    @test getindepsym_defaultt(A) == :t

    vars = interpret_vars(nothing, A)
    @test vars == [(DEFAULT_PLOT_FUNC, 0, 1), (DEFAULT_PLOT_FUNC, 0, 2)]

    labels = String[]
    @test add_labels!(labels, vars[1], 2, A, ["t", "u[1]"]) === labels
    @test labels == ["u[1]"]

    plot_vecs, plot_labels = diffeq_to_arrays(
        A, false, 100, nothing, vars, :identity, nothing
    )
    @test plot_vecs == [[0.0 0.0; 1.0 1.0], [1.0 2.0; 3.0 4.0]]
    @test plot_labels == ["u[1]", "u[2]"]
end

# Regression for DifferentialEquations.jl#360: input-arity of idxs specs is not
# plot dimensionality. Mixing a bare index / (0,i) with (f,0,i,j) when f returns
# a 2-tuple must succeed; mixing a 2-D series with a 3-D series must throw.
# Labels for custom transforms must stay `f(...)` (not the bare last index),
# and dims must come from evaluated output (no probe with integer 1s).
@testset "plot idxs with mixed input arity but matching output dims (#360)" begin
    t = [0.0, 0.25, 0.5, 0.75, 1.0]
    u = [[1.0, 2.0, 10.0 + tt, 20.0 + 2tt] for tt in t]
    A = DiffEqArray(u, t)

    adder(tt, a, b) = (tt, a + b)
    adder3(tt, a, b) = (tt, a, b)
    g(tt::Float64, x::Float64) = (tt, x)
    dom(tt, x) = (tt, sqrt(x - 2))

    function plot_sparse(idxs)
        vars = interpret_vars(idxs, A)
        return diffeq_to_arrays(A, false, 100, nothing, vars, :identity, nothing)
    end

    u3 = [uu[3] for uu in u]
    u3pu4 = [uu[3] + uu[4] for uu in u]

    for idxs in ([3, (adder, 0, 3, 4)], [(0, 3), (adder, 0, 3, 4)])
        plot_vecs, labels = plot_sparse(idxs)
        @test length(plot_vecs) == 2
        @test size(plot_vecs[1], 2) == 2
        @test plot_vecs[1][:, 1] ≈ t
        @test plot_vecs[2][:, 1] ≈ u3
        @test plot_vecs[1][:, 2] ≈ t
        @test plot_vecs[2][:, 2] ≈ u3pu4
        @test labels == ["u[3]", "f(t,u[3],u[4])"]
    end

    _, labels01 = plot_sparse((0, 1))
    @test labels01 == ["u[1]"]

    plot_vecs_g, labels_g = plot_sparse([(g, 0, 3)])
    @test plot_vecs_g[1][:, 1] ≈ t
    @test plot_vecs_g[2][:, 1] ≈ u3
    @test labels_g == ["f(t,u[3])"]

    plot_vecs_dom, labels_dom = plot_sparse([(dom, 0, 3)])
    @test plot_vecs_dom[1][:, 1] ≈ t
    @test plot_vecs_dom[2][:, 1] ≈ sqrt.(u3 .- 2)
    @test labels_dom == ["f(t,u[3])"]

    err = try
        plot_sparse([3, (adder3, 0, 3, 4)])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("output dimension", sprint(showerror, err))
end
