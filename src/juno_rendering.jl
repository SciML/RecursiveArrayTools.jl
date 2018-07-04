@require Juno="e5e0dc1b-0480-54bc-9374-aad01c23163d" begin
  # Juno Rendering
  Juno.render(i::Juno.Inline, x::ArrayPartition) = Juno.render(i, Juno.defaultrepr(x))
  Juno.render(i::Juno.Inline, x::AbstractVectorOfArray) = Juno.render(i, Juno.defaultrepr(x))
end
