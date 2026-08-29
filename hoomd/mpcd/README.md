# MPCD

`hoomd.mpcd` implements the multiparticle collision dynamics method for
simulating fluctuating hydrodynamic interactions. Because MPCD typically uses a
large number of solvent particles, separate a separate particle data structure
is provided.

## Citing MPCD

When using `hoomd.mpcd` in published work, please cite:

* M.P. Howard, A.Z. Panagiotopoulos, and A. Nikoubashman. "Efficient mesoscale
  hydrodynamics: Multiparticle collision dynamics with massively parallel GPU
  acceleration." *Computer Physics Communications* **230**, 10-20 (2018).
  [10.1016/j.cpc.2018.04.009](https://doi.org/10.1016/j.cpc.2018.04.009).

When using rigid bodies with `hoomd.mpcd`, please also cite:

* M. Bush, J.C. Palmer, and M.P. Howard. "Simulating hydrodynamic interactions
  in colloidal suspensions using multiparticle collision dynamics with
  rigid-body constraints." *The Journal of Chemical Physics* **165**, 044904
  (2026). [10.1063/5.0339394](https://doi.org/10.1063/5.0339394).

## Acknowledgements

We gratefully acknowledge the following support for the development of
`hoomd.mpcd` and its associated documentation and tutorials:

* The initial development of `hoomd.mpcd`, released in version 2.3.0, was part
  of the Blue Waters sustained-petascale computing project, which was supported
  by the National Science Foundation under Award Nos. 0725070 and 1238993 and
  by the state of Illinois.

* The expanded development of `hoomd.mpcd`, beginning in version 4.8.0, was
  supported by the National Science Foundation under Award Nos. 2310724 and
  2310725.

Any opinions, findings and conclusions or recommendations expressed in this
material are those of the author(s) and do not necessarily reflect the views of
the National Science Foundation.
