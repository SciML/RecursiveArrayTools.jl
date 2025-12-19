using Pkg
Pkg.add("JET")
Pkg.instantiate()
using JET
using RecursiveArrayTools

# Get all reports first
result = JET.report_package(RecursiveArrayTools; target_modules = (RecursiveArrayTools,))
reports = JET.get_reports(result)

# Filter out similar_type inference errors from StaticArraysCore
filtered_reports = filter(reports) do report
    s = string(report)
    !(occursin("similar_type", s) && occursin("StaticArraysCore", s))
end

# Check if there are any non-filtered errors
@test isempty(filtered_reports)
