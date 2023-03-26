# Define a new species of projection operator for this type:
ChainRulesCore.ProjectTo(x::VectorOfArray) = ChainRulesCore.ProjectTo{VectorOfArray}()
