module RecursiveArrayToolsStructArraysExt

import RecursiveArrayTools, StructArrays
RecursiveArrayTools.rewrap(::StructArrays.StructArray, u) = StructArrays.StructArray(u)

end