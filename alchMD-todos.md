This file is used for logging design decisions and unimplmeneted choices in the alchMD branch so that they can be discussed as part of the pull request rather than making issues.

# TODO
[x] Write default PotentialPair match for signatures without alchemy bool
[] Fail gracefully with XPLOR on
[X] generalize the extra functions in PotentialPair?
[X] Figure out array access with derived class specialized add-ins

# Design choices
- When an alchemical particle is disabled, it must rewrite the associated parameters.
- When an alchemy particle is re-enabled it needs to implement the same constructor logic.

# Incomplete/Removed for Simplicity
- Saving alchemical particle information as part of snapshot data/system definition.
- Normalization protocall is currently not included
- MPI compatibility (lower priority, figure out how to disable for now)
- GPU functionality
- per particle access to alchemical forces

# Ownership
- Alchemical Force Computes
    - M_alpha (max number of alchemical variables implemented)
    - alchemy_used: Boolean array, shape M_types x M_types x M_alpha (should probably be renamed)
    - Parameters original values
    - 1D arrays matching number and order of trues in used, length matches 

- Alchemical Particle
    - Alchemical Position (currently implemented in dimensionless alpha space
    - Alchemical Kinetic Variables
        - Velocity
        - Mass
        - Alchemical Potential (mu_alpha)
        - Net Alchemical Force (cheap to save and don't have to worry about recomputing if needed unexpectedly for loggin etc)
    - Associated alchemical force compute
    - Associated alchemostat (just to make sure we only are using one per particle)

- Alchemostat
    - Alchemical Temperature
    - Alchemical Timestep
    - List of alchemical forces to integrate

# Future enhancements
- External Biases
- Dynamically update rcut
- Helper functions 
