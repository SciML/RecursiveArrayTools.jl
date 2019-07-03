function __init__()
  @require Juno="e5e0dc1b-0480-54bc-9374-aad01c23163d" begin
    # Juno Rendering
    Juno.render(i::Juno.Inline, x::ArrayPartition) = Juno.render(i, Juno.defaultrepr(x))
    Juno.render(i::Juno.Inline, x::AbstractVectorOfArray) = Juno.render(i, Juno.defaultrepr(x))
  end

  @require ApproxFun="28f2ccd6-bb30-5033-b560-165f7b14dc2f" begin
    RecursiveArrayTools.recursive_unitless_eltype(a::ApproxFun.Fun) = typeof(a)
    RecursiveArrayTools.recursive_unitless_bottom_eltype(a::ApproxFun.Fun) = recursive_unitless_bottom_eltype(ApproxFun.coefficients(a))
    RecursiveArrayTools.recursive_bottom_eltype(a::ApproxFun.Fun) = recursive_bottom_eltype(ApproxFun.coefficients(a))
  end
end
