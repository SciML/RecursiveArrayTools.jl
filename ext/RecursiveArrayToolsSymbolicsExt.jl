module RecursiveArrayToolsSymbolicsExt

import RecursiveArrayTools
isdefined(Base, :get_extension) ? (import Symbolics) : (import ..Symbolics)

RecursiveArrayTools.issymbollike(::Union{Symbolics.BasicSymbolic,Symbolics.Num}) = true

end
