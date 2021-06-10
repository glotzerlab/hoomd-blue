This file is used for logging design decisions and unimplmeneted choices in the
alchMD branch so that they can be discussed as part of the pull request rather
than making issues.

# TODO
- [x] Write default PotentialPair match for signatures without alchemy bool
- [ ] Logging and data access
- [ ] Fail gracefully with XPLOR on
- [X] generalize the extra functions in PotentialPair?
- [X] Figure out array access with derived class specialized add-ins
- [ ] Decide where the order of the alchemical parameters will be stored
  -  std::vector<std::string> in eval?
  -  would be best to only check during compile time or through some test rather
     than take up space
- [X] Enable alchemostat to be called directly before and directly after
  canonical thermostat
  - Reverse second half step call order for integrators
  - Require alchemostat first
- [ ] Deal with the alchemostat having to be first in the hoomd.context.current.
  integration_methods
  -  generally true for Trotter factorization, user would be resposible if more
     than 2 timescales were used
- [ ] Alchmostat
  - [ ] Issues with empty group and methods for base integrators getting called?
  - [ ] Allow for alchemical temperature to be specified as a fraction of
    canonical temperature
- [ ] Robust next alchem timestep logic
  - [ ] Flowing from the integrators
  - [ ] Flexible to cover multiple integrators accesssing the same
    pairinteraction base, min? but then would miss nexts, list/set
- [ ] Alchemical Mass logic
  -  Full partition function in a fully coupled system would dictate a Beta value
  -  To handle different initial conditions in the same parameter space, can
    be useful to set it as inverse squared initial
- [ ] implement reflection etc for constraining alpha to positive values?



# Incomplete/Removed for Simplicity
- Saving alchemical particle information as part of snapshot data/system definition.
- Normalization protocol is currently not included
- XPLOR compatibility
- MPI compatibility (lower priority, figure out how to disable for now)
- GPU functionality
- per particle access to alchemical forces
- Delay the initial alchemical timestep (usually helpful for initialization)

## Future enhancements
- External Biases
- Dynamically update rcut
- Helper functions

## Tests
- Alchemostat without thermostat actually computes alchemical forces
- Alchemical time factor 0 causes error
-

# Design choices
- When an alchemical particle is disabled, it must rewrite the associated parameters.
- When an alchemy particle is re-enabled it needs to implement the same constructor logic.
- Alchemostat must be the first in the list of integrators to allow for the
  separation of timescales
- Store a pair (timestep,averaged netforce) when the compute is run in the
  alchemical data, should be able to use it to handle some cases where the time
  factor is changed
-
## Ownership
- Alchemical Force Computes
    - M_alpha (max number of alchemical variables implemented)
    - alchemy_used: Boolean array, shape M_types x M_types x M_alpha (should probably be renamed)
    - Parameters original values
    - 1D arrays matching number and order of trues in used, length matches

- Alchemical Particle/ Alchemical Data
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
