using RecursiveArrayTools, Symbolics, Test

@variables x;
@test RecursiveArrayTools.issymbollike(x)
