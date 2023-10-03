module RecursiveArrayToolsSymbolicsExt

import RecursiveArrayTools
isdefined(Base, :get_extension) ? (import Symbolics) : (import ..Symbolics)

RecursiveArrayTools.issymbollike(::Symbolics.Num) = true

end
