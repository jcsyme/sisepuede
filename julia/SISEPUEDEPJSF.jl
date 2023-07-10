"""
Get some functions to support interaction with Julia through Python
"""
module SISEPUEDEPJSF

using Pkg

export check_solvers

# rank solver packages--searches for replacement if specified solver is unavailable
rank_ordered_solver_list = Vector{String}([
    "CPLEX",
    "Gurobi",
    "GAMS",
    "HiGHS",
    "Clp",
    "Cbc",
    "GLPK"
])


"""
Check for solver package if avaiable; return best available solver if specified is unavailable.

##  Constructs

```
function check_solvers(
    solver_in::String
)::nothing
```

##  Function Arguments
- `solver_in`: string specifying solver package to check for (case-sensitive)

##  Keyword Arguments
- `solver_rank_list`: ordered list specifying the rank of each alternative solver if `solver_in` is unavailable
"""
function check_solvers(
    solver_in::String,
    solver_rank_list::Vector{String} = rank_ordered_solver_list
)::Union{String, Nothing}
    # get package dependencies
    pkg_dependencies = collect(keys(Pkg.project().dependencies))
    n_solvers = length(solver_rank_list)

    if solver_in in pkg_dependencies
        return solver_in
    else
        i = 1
        while !(solver_rank_list[i] in pkg_dependencies) & (i <= n_solvers)
            i += 1
        end

        return (i > n_solvers) ? Nothing : solver_rank_list[i]
    end
end


# end module
end
