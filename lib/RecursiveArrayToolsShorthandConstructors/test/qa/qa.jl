using SciMLTesting, RecursiveArrayToolsShorthandConstructors, JET, Test

const RATSC = RecursiveArrayToolsShorthandConstructors

run_qa(
    RATSC;
    explicit_imports = true,
    # `getindex(::Type{VA}, ...)` / `getindex(::Type{AP}, ...)` extend Base on
    # RecursiveArrayTools-owned types, so they are intentional (owned) methods,
    # not piracy.
    aqua_kwargs = (; piracies = (; treat_as_own = [RATSC.VA, RATSC.AP])),
    jet_kwargs = (; target_defined_modules = true),
)
