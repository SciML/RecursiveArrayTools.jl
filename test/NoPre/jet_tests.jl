using JET, Test, RecursiveArrayTools

# Get all reports first
result = JET.report_package(RecursiveArrayTools; target_modules = (RecursiveArrayTools,))
reports = JET.get_reports(result)

# Filter out known false positives
filtered_reports = filter(reports) do report
    s = string(report)
    # StaticArraysCore similar_type inference
    occursin("similar_type", s) && occursin("StaticArraysCore", s) && return false
    # RecipesBase user recipe keywords (denseplot, plotdensity, etc.) are dynamic
    occursin("is_key_supported", s) && occursin("RecipesBase", s) && return false
    return true
end

# Check if there are any non-filtered errors
@test isempty(filtered_reports)
