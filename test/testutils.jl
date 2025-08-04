using RecursiveArrayTools
using Tables
using Tables: IteratorInterfaceExtensions

# Test Tables interface with row access + IteratorInterfaceExtensions for QueryVerse
# (see https://tables.juliadata.org/stable/#Testing-Tables.jl-Implementations)
function test_tables_interface(x::AbstractDiffEqArray, names::Vector{Symbol},
        values::Matrix)
    @assert length(names) == size(values, 2)

    # AbstractDiffEqArray is a table with row access
    @test Tables.istable(x)
    @test Tables.istable(typeof(x))
    @test Tables.rowaccess(x)
    @test Tables.rowaccess(typeof(x))
    @test !Tables.columnaccess(x)
    @test !Tables.columnaccess(typeof(x))

    # Check implementation of AbstractRow iterator
    tbl = Tables.rows(x)
    @test length(tbl) == size(values, 1)
    @test Tables.istable(tbl)
    @test Tables.istable(typeof(tbl))
    @test Tables.rowaccess(tbl)
    @test Tables.rowaccess(typeof(tbl))
    @test Tables.rows(tbl) === tbl

    # Check implementation of AbstractRow subtype
    for (i, row) in enumerate(tbl)
        @test eltype(tbl) === typeof(row)
        @test propertynames(row) == Tables.columnnames(row) == names
        for (j, name) in enumerate(names)
            @test getproperty(row, name) == Tables.getcolumn(row, name) ==
                  Tables.getcolumn(row, j) == values[i, j]
        end
    end

    # Check column access
    coltbl = Tables.columns(x)
    @test length(coltbl) == size(values, 2)
    @test Tables.istable(coltbl)
    @test Tables.istable(typeof(coltbl))
    @test Tables.columnaccess(coltbl)
    @test Tables.columnaccess(typeof(coltbl))
    @test Tables.columns(coltbl) === coltbl
    @test propertynames(coltbl) == Tables.columnnames(coltbl) == Tuple(names)
    for (i, name) in enumerate(names)
        @test getproperty(coltbl, name) == Tables.getcolumn(coltbl, name) ==
              Tables.getcolumn(coltbl, i) == values[:, i]
    end

    # IteratorInterfaceExtensions
    @test IteratorInterfaceExtensions.isiterable(x)
    iterator = IteratorInterfaceExtensions.getiterator(x)
    for (i, row) in enumerate(iterator)
        @test row isa NamedTuple
        @test propertynames(row) == Tuple(names)
        for (j, name) in enumerate(names)
            @test getproperty(row, name) == row[j] == values[i, j]
        end
    end

    nothing
end
