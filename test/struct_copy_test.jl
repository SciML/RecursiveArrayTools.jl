using RecursiveArrayTools, Test

# Test structures for struct-aware recursivecopy
struct SimpleStruct
    a::Int
    b::Float64
end

mutable struct MutableStruct
    a::Vector{Float64}
    b::Matrix{Int}
    c::String
end

struct NestedStruct
    simple::SimpleStruct
    mutable::MutableStruct
    array::Vector{Int}
end

struct ParametricStruct{T}
    data::Vector{T}
    metadata::T
end

@testset "Struct recursivecopy tests" begin
    
    @testset "Simple immutable struct" begin
        original = SimpleStruct(42, 3.14)
        copied = recursivecopy(original)
        
        @test copied isa SimpleStruct
        @test copied.a == original.a
        @test copied.b == original.b
        # Note: For immutable structs with only primitive types, Julia may optimize 
        # to use the same memory location, so we test functionality rather than identity
    end
    
    @testset "Mutable struct with arrays" begin
        original = MutableStruct([1.0, 2.0, 3.0], [1 2; 3 4], "test")
        
        # Should error for mutable structs
        @test_throws ErrorException("recursivecopy for mutable structs is not currently implemented. Use deepcopy instead.") recursivecopy(original)
    end
    
    @testset "Nested struct" begin
        simple = SimpleStruct(10, 2.5)
        # Create a nested struct with only immutable components
        struct ImmutableNested
            simple::SimpleStruct
            array::Vector{Int}
            name::String
        end
        
        original = ImmutableNested(simple, [100, 200, 300], "nested")
        copied = recursivecopy(original)
        
        @test copied isa ImmutableNested
        @test copied.simple.a == original.simple.a
        @test copied.simple.b == original.simple.b
        @test copied.array == original.array
        @test copied.name == original.name
        
        @test copied !== original
        @test copied.array !== original.array
        
        # Test independence
        original.array[1] = 999
        @test copied.array[1] == 100  # Should remain unchanged
    end
    
    @testset "Parametric struct" begin
        original = ParametricStruct([1, 2, 3], 42)
        copied = recursivecopy(original)
        
        @test copied isa ParametricStruct{Int}
        @test copied.data == original.data
        @test copied.metadata == original.metadata
        @test copied !== original
        @test copied.data !== original.data
    end
    
    @testset "Compatibility with existing types" begin
        # Test that arrays still work
        arr = [1, 2, 3]
        copied_arr = recursivecopy(arr)
        @test copied_arr == arr
        @test copied_arr !== arr
        
        # Test that numbers still work
        num = 42
        copied_num = recursivecopy(num)
        @test copied_num == num
        
        # Test that strings still work
        str = "hello"
        copied_str = recursivecopy(str)
        @test copied_str == str
    end
    
    @testset "ArrayPartition with structs" begin
        simple1 = SimpleStruct(1, 1.0)
        simple2 = SimpleStruct(2, 2.0)
        ap = ArrayPartition([simple1, simple2])
        copied_ap = recursivecopy(ap)
        
        @test copied_ap isa ArrayPartition
        @test length(copied_ap.x) == length(ap.x)
        @test copied_ap.x[1][1].a == ap.x[1][1].a
        @test copied_ap !== ap
        @test copied_ap.x[1] !== ap.x[1]
    end
    
    @testset "Array dispatch still works correctly" begin
        # Test that our struct method doesn't interfere with existing array methods
        
        # Arrays of numbers should use copy
        num_array = [1, 2, 3]
        copied_num = recursivecopy(num_array)
        @test copied_num == num_array
        @test copied_num !== num_array
        
        # Arrays of arrays should recursively copy
        nested_array = [[1, 2], [3, 4]]
        copied_nested = recursivecopy(nested_array)
        @test copied_nested == nested_array
        @test copied_nested !== nested_array
        @test copied_nested[1] !== nested_array[1]
        @test copied_nested[2] !== nested_array[2]
        
        # AbstractVectorOfArray should use its method
        ap = ArrayPartition([1.0, 2.0], [3, 4])
        copied_ap = recursivecopy(ap)
        @test copied_ap isa ArrayPartition
        @test copied_ap.x[1] == ap.x[1]
        @test copied_ap.x[1] !== ap.x[1]
        
        # Test that structs containing arrays still work
        struct StructWithArrays
            data::Vector{Vector{Int}}
            metadata::String
        end
        
        original_struct = StructWithArrays([[1, 2], [3, 4]], "test")
        copied_struct = recursivecopy(original_struct)
        @test copied_struct.data == original_struct.data
        @test copied_struct.data !== original_struct.data
        @test copied_struct.data[1] !== original_struct.data[1]
        @test copied_struct.metadata == original_struct.metadata
    end
end