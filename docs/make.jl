using Documenter, RecursiveArrayTools

include("pages.jl")

makedocs(
    sitename="RecursiveArrayTools.jl",
    authors="Chris Rackauckas",
    modules=[RecursiveArrayTools],
    clean=true,doctest=false,
    format = Documenter.HTML(analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://recursivearraytools.sciml.ai/stable/"),
    pages=pages
)

deploydocs(
   repo = "github.com/SciML/RecursiveArrayTools.jl.git";
   push_preview = true
)
