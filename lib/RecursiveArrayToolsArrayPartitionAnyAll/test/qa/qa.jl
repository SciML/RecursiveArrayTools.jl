using SciMLTesting, RecursiveArrayToolsArrayPartitionAnyAll, JET, Test

const RATAPAA = RecursiveArrayToolsArrayPartitionAnyAll

run_qa(
    RATAPAA;
    explicit_imports = true,
    # `any`/`all` are extended on the RecursiveArrayTools-owned `ArrayPartition`
    # type, so they are intentional (owned) methods, not piracy.
    aqua_kwargs = (; piracies = (; treat_as_own = [RATAPAA.ArrayPartition])),
    jet_kwargs = (; target_defined_modules = true),
)
