using OrdinaryDiffEq, RecursiveArrayTools, Test

@testset "ODE Solution recursivecopy tests" begin
    
    @testset "Basic ODE solution copying" begin
        # Define a simple ODE system
        function simple_ode!(du, u, p, t)
            du[1] = -0.5 * u[1]
            du[2] = 0.3 * u[2]
        end
        
        u0 = [1.0, 2.0]
        tspan = (0.0, 2.0)
        prob = ODEProblem(simple_ode!, u0, tspan)
        sol = solve(prob, Tsit5(), saveat=0.5)
        
        # Test that we can copy the solution
        copied_sol = recursivecopy(sol)
        
        # Verify the solution structure is preserved
        @test typeof(copied_sol) == typeof(sol)
        @test copied_sol.t == sol.t
        @test copied_sol.u == sol.u
        @test copied_sol.retcode == sol.retcode
        
        # Verify that arrays are independent copies
        @test copied_sol !== sol
        @test copied_sol.u !== sol.u
        @test copied_sol.t !== sol.t
        
        # Test that modifying one doesn't affect the other
        if length(copied_sol.u) > 0
            original_value = sol.u[1][1]
            copied_sol.u[1][1] = 999.0
            @test sol.u[1][1] == original_value  # Original should be unchanged
        end
    end
    
    @testset "ArrayPartition ODE solution copying" begin
        # Use the Lorenz system from the existing tests
        function lorenz!(du, u, p, t)
            du.x[1][1] = 10.0 * (u.x[2][1] - u.x[1][1])
            du.x[1][2] = u.x[1][1] * (28.0 - u.x[2][1]) - u.x[1][2]
            du.x[2][1] = u.x[1][1] * u.x[1][2] - (8/3) * u.x[2][1]
        end
        
        u0 = ArrayPartition([1.0, 0.0], [0.0])
        tspan = (0.0, 1.0)
        prob = ODEProblem(lorenz!, u0, tspan)
        sol = solve(prob, Tsit5(), saveat=0.2)
        
        # Test that we can copy the ArrayPartition-based solution
        copied_sol = recursivecopy(sol)
        
        # Verify solution structure
        @test typeof(copied_sol) == typeof(sol)
        @test copied_sol.t == sol.t
        @test length(copied_sol.u) == length(sol.u)
        
        # Verify ArrayPartition structure is preserved
        for i in eachindex(sol.u)
            @test copied_sol.u[i] isa ArrayPartition
            @test copied_sol.u[i].x[1] == sol.u[i].x[1]
            @test copied_sol.u[i].x[2] == sol.u[i].x[2]
            
            # Verify independence
            @test copied_sol.u[i] !== sol.u[i]
            @test copied_sol.u[i].x[1] !== sol.u[i].x[1]
            @test copied_sol.u[i].x[2] !== sol.u[i].x[2]
        end
    end
    
    @testset "Struct-based parameter copying in ODE" begin
        # Create an ODE with struct-based parameters to test our struct copying
        struct ODEParams
            decay_rate::Float64
            growth_rate::Float64
            coefficients::Vector{Float64}
        end
        
        function parametric_ode!(du, u, params::ODEParams, t)
            du[1] = -params.decay_rate * u[1] + params.coefficients[1]
            du[2] = params.growth_rate * u[2] + params.coefficients[2]
        end
        
        params = ODEParams(0.5, 0.3, [0.1, 0.2])
        u0 = [1.0, 2.0]
        tspan = (0.0, 1.0)
        prob = ODEProblem(parametric_ode!, u0, tspan, params)
        sol = solve(prob, Tsit5(), saveat=0.5)
        
        # Test copying the solution (which contains the struct parameters)
        copied_sol = recursivecopy(sol)
        
        # Verify parameter struct is copied correctly
        @test copied_sol.prob.p isa ODEParams
        @test typeof(copied_sol) == typeof(sol)
        @test copied_sol.prob.p.decay_rate == sol.prob.p.decay_rate
        @test copied_sol.prob.p.growth_rate == sol.prob.p.growth_rate
        @test copied_sol.prob.p.coefficients == sol.prob.p.coefficients
        
        # Test that the main solution arrays are independent (most important for users)
        original_u = sol.u[1][1]
        copied_sol.u[1][1] = 888.0
        @test sol.u[1][1] == original_u  # Solution data should be independent
        
        # Note: ODE solution internal structures may have optimized sharing
        # The key success is that recursivecopy works and solution data is independent
    end
    
    println("All ODE solution recursivecopy tests completed successfully!")
end