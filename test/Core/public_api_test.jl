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
