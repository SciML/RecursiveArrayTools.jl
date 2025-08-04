module RecursiveArrayToolsIteratorInterfaceExtensionsExt

import RecursiveArrayTools: AbstractDiffEqArray
import IteratorInterfaceExtensions

# Iterator interface for QueryVerse
# (see also https://tables.juliadata.org/stable/#Tables.datavaluerows)
IteratorInterfaceExtensions.isiterable(::AbstractDiffEqArray) = true

# Note: getiterator functionality requires Tables.jl and will be available
# when both Tables.jl and IteratorInterfaceExtensions.jl are loaded
# via the TablesExt extension.

end
