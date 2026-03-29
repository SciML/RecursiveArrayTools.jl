# Plotting

`AbstractVectorOfArray` and `AbstractDiffEqArray` types include
[Plots.jl](https://github.com/JuliaPlots/Plots.jl) recipes so they can be
visualised directly with `plot(A)`.

## VectorOfArray

A `VectorOfArray` plots as a matrix where each inner array is a column:

```julia
using RecursiveArrayTools, Plots
A = VectorOfArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
plot(A)   # 3 series (components) vs column index
```

## DiffEqArray

A `DiffEqArray` plots as component time series against `A.t`:

```julia
u = [[sin(t), cos(t)] for t in 0:0.1:2pi]
t = collect(0:0.1:2pi)
A = DiffEqArray(u, t)
plot(A)   # plots sin and cos vs t
```

If the `DiffEqArray` carries a symbolic system (via `variables` and
`independent_variables` keyword arguments), the axis labels and series names
are set automatically from the symbol names.

## Dense (Interpolated) Plotting

When a `DiffEqArray` has an interpolation object in its `interp` field and
`dense = true`, calling `plot(A)` generates a smooth curve by evaluating the
interpolation at many points rather than connecting only the saved time steps.

```julia
using SciMLBase: LinearInterpolation

u = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
t = [0.0, 1.0, 2.0, 3.0]
interp = LinearInterpolation(t, u)
A = DiffEqArray(u, t; interp = interp, dense = true)
plot(A)   # smooth interpolated curve with 1000+ points
```

### Plot Recipe Keyword Arguments

The `AbstractDiffEqArray` recipe accepts the following keyword arguments,
which can be passed directly to `plot`:

| Keyword | Type | Default | Description |
|---------|------|---------|-------------|
| `denseplot` | `Bool` | `A.dense && A.interp !== nothing` | Use dense interpolation for smooth curves. Set `false` to show only saved points. |
| `plotdensity` | `Int` | `max(1000, 10 * length(A.u))` | Number of evenly-spaced points to evaluate when `denseplot = true`. |
| `tspan` | `Tuple` or `nothing` | `nothing` | Restrict the time window. E.g. `tspan = (0.0, 5.0)`. |
| `idxs` | varies | `nothing` | Select which components to plot (see below). |

Example:

```julia
plot(A; denseplot = true, plotdensity = 5000, tspan = (0.0, 2.0))
```

## Selecting Variables with `idxs`

The `idxs` keyword controls which variables appear in the plot. It supports
several formats:

### Single component

```julia
plot(A; idxs = 1)      # plot component 1 vs time
plot(A; idxs = 2)      # plot component 2 vs time
```

### Multiple components

```julia
plot(A; idxs = [1, 3, 5])   # plot components 1, 3, 5 vs time
```

### Phase-space plots (component vs component)

Use a tuple where index `0` represents the independent variable (time):

```julia
plot(A; idxs = (1, 2))      # component 1 vs component 2
plot(A; idxs = (0, 1))      # time vs component 1 (same as default)
plot(A; idxs = (1, 2, 3))   # 3D plot of components 1, 2, 3
```

### Symbolic indexing

When the `DiffEqArray` carries a symbolic system, variables can be referenced
by symbol:

```julia
A = DiffEqArray(u, t; variables = [:x, :y], independent_variables = [:t])
plot(A; idxs = :x)           # plot x vs time
plot(A; idxs = [:x, :y])     # plot both
plot(A; idxs = (:x, :y))     # phase plot of x vs y
```

### Custom transformations

A function can be applied to the plotted values:

```julia
plot(A; idxs = (norm, 0, 1, 2))   # plot norm(u1, u2) vs time
```

The tuple format is `(f, xvar, yvar)` or `(f, xvar, yvar, zvar)` where `f`
is applied element-wise.

## Callable Interface

Any `AbstractDiffEqArray` with an `interp` field supports callable syntax for
interpolation, independent of plotting:

```julia
A(0.5)                    # interpolate all components at t=0.5
A(0.5; idxs = 1)          # interpolate component 1 at t=0.5
A([0.1, 0.5, 0.9])       # interpolate at multiple times (returns DiffEqArray)
A(0.5, Val{1})            # first derivative at t=0.5
A(0.5; continuity = :right)  # right-continuity at discontinuities
```

The interpolation object must be callable as
`interp(t, idxs, deriv, p, continuity)`, matching the protocol used by
SciMLBase's `LinearInterpolation`, `HermiteInterpolation`, and
`ConstantInterpolation`.

When no interpolation is available (`interp === nothing`), calling `A(t)`
throws an error.

## ODE Solution Plotting

ODE solutions from DifferentialEquations.jl are subtypes of
`AbstractDiffEqArray` and inherit all of the above functionality, plus
additional features:

- **Automatic dense plotting**: `denseplot` defaults to `true` when the solver
  provides dense output.
- **Analytic solution overlay**: `plot(sol; plot_analytic = true)` overlays the
  exact solution if the problem defines one.
- **Discrete variables**: Time-varying parameters are plotted as step functions
  with dashed lines and markers.
- **Symbolic indexing**: Full symbolic indexing through ModelingToolkit is
  supported, including observed (derived) variables.

These advanced features are defined in SciMLBase and activate automatically
when plotting solution objects.
