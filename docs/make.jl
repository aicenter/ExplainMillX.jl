using Documenter
using ExplainMillX
using Literate

# The tutorial is examples/mutagenesis.jl itself -- Literate.jl renders it
# into a Documenter page (executing it for real) so the tutorial and the
# runnable example can never drift apart. Matches JsonGrinder.jl's own
# tutorial convention.
const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const GENERATED_DIR = joinpath(@__DIR__, "src", "generated")

Literate.markdown(
    joinpath(EXAMPLES_DIR, "mutagenesis.jl"),
    GENERATED_DIR;
    documenter=true,
    execute=true,
)

DocMeta.setdocmeta!(ExplainMillX, :DocTestSetup, :(using ExplainMillX); recursive=true)

makedocs(;
    modules=[ExplainMillX],
    sitename="ExplainMillX.jl",
    authors="Pevnak",
    # No git remote is configured yet (`git remote -v` is empty), so there's
    # nothing to link "Edit on GitHub" source buttons to. Set explicitly
    # once the repository has a real remote; Documenter would otherwise
    # error trying to auto-detect one.
    remotes=nothing,
    pages=[
        "Home" => "index.md",
        "Tutorial: Mutagenesis" => "generated/mutagenesis.md",
        "How-to Guides" => [
            "howto/explain_prediction.md",
            "howto/explain_nonsoftmax.md",
            "howto/json_output.md",
            "howto/choose_pruning_strategy.md",
            "howto/reproducibility.md",
            "howto/performance.md",
            "howto/inspect_mask.md",
        ],
        "Explanation" => [
            "explanation/what_is_explanation.md",
            "explanation/scoring_and_pruning.md",
            "explanation/confidence_gap.md",
            "explanation/hierarchy_and_masks.md",
            "explanation/non_uniqueness.md",
            "explanation/limitations.md",
        ],
        "API Reference" => "reference.md",
    ],
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
    ),
    checkdocs=:exports,
)

# `deploydocs` is intentionally not wired up yet: it needs a real GitHub
# remote (`git remote -v` is currently empty) and a CI workflow to run it
# from. Add once the repository has a home:
#
# deploydocs(;
#     repo="github.com/<org>/ExplainMillX.jl",
#     devbranch="main",
# )
