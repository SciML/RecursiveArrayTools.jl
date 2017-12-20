@require Juno begin
  # Juno Rendering
  Juno.render(i::Juno.Inline, x::ArrayPartition) = Juno.render(i, Juno.defaultrepr(x))
  Juno.render(i::Juno.Inline, x::AbstractVectorOfArray) = Juno.render(i, Juno.defaultrepr(x))
end
