[![HOOMD-blue](sphinx-doc/hoomdblue-logo-horizontal.svg)](https://glotzerlab.engin.umich.edu/hoomd-blue/)

[![Citing HOOMD](https://img.shields.io/badge/cite-hoomd-blue.svg)](https://hoomd-blue.readthedocs.io/en/latest/citing.html)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/hoomd.svg?style=flat)](https://anaconda.org/conda-forge/hoomd)
[![conda-forge Downloads](https://img.shields.io/conda/dn/conda-forge/hoomd.svg?style=flat)](https://anaconda.org/conda-forge/hoomd)
[![GitHub Actions](https://github.com/glotzerlab/hoomd-blue/actions/workflows//test.yml/badge.svg?branch=trunk-patch)](https://github.com/glotzerlab/hoomd-blue/actions/workflows/test.yml)
[![Read the Docs](https://img.shields.io/readthedocs/hoomd-blue/latest.svg)](https://hoomd-blue.readthedocs.io/en/latest/?badge=latest)
[![Contributors](https://img.shields.io/github/contributors-anon/glotzerlab/hoomd-blue.svg?style=flat)](https://hoomd-blue.readthedocs.io/en/latest/credits.html)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

**HOOMD-blue** is a Python package that runs simulations of particle systems on CPUs and GPUs. It
performs hard particle Monte Carlo simulations of a variety of shape classes and molecular dynamics
simulations of particles with a range of pair, bond, angle, and other potentials. Many features are
targeted at the soft matter research community, though the code is general and capable of many
types of particle simulations.

## Resources

- [Documentation](https://hoomd-blue.readthedocs.io/):
  Tutorial, full Python API description, and usage information.
- [Citing HOOMD-blue](https://hoomd-blue.readthedocs.io/en/latest/citing.html)
  How to cite the code.
- [Installation guide](INSTALLING.rst):
  Instructions for installing **HOOMD-blue** binaries.
- [Compilation guide](BUILDING.rst):
  Instructions for compiling **HOOMD-blue**.
- [HOOMD-blue discussion board](https://github.com/glotzerlab/hoomd-blue/discussions/):
  Ask the **HOOMD-blue** user community for help.
- [HOOMD-blue website](https://glotzerlab.engin.umich.edu/hoomd-blue/):
  Additional information and publications.
- [HOOMD-blue benchmark scripts](https://github.com/glotzerlab/hoomd-benchmarks):
  Scripts to evaluate the performance of HOOMD-blue simulations.
- [HOOMD-blue validation tests](https://github.com/glotzerlab/hoomd-validation):
  Scripts to validate that HOOMD-blue performs accurate simulations.

## Related tools

- [freud](https://freud.readthedocs.io/):
  Analyze HOOMD-blue simulation results with the **freud** Python library.
- [signac](https://signac.io/):
  Manage your workflow with **signac**.

## Example scripts

These examples demonstrate some of the Python API.

Hard particle Monte Carlo:
```python
import hoomd

mc = hoomd.hpmc.integrate.ConvexPolyhedron()
mc.shape['octahedron'] = dict(vertices=[
    (-0.5, 0, 0),
    (0.5, 0, 0),
    (0, -0.5, 0),
    (0, 0.5, 0),
    (0, 0, -0.5),
    (0, 0, 0.5),
])

cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu, seed=20)
sim.operations.integrator = mc
# The tutorial describes how to construct an initial configuration 'init.gsd'.
sim.create_state_from_gsd(filename='init.gsd')

sim.run(1e5)
```

Molecular dynamics:
```python
import hoomd

cell = hoomd.md.nlist.Cell(buffer=0.4)
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
lj.r_cut[('A', 'A')] = 2.5

integrator = hoomd.md.Integrator(dt=0.005)
integrator.forces.append(lj)
bussi = hoomd.md.methods.thermostats.Bussi(kT=1.5)
nvt = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All(), thermostat=bussi)
integrator.methods.append(nvt)

gpu = hoomd.device.GPU()
sim = hoomd.Simulation(device=gpu)
sim.operations.integrator = integrator
# The tutorial describes how to construct an initial configuration 'init.gsd'.
sim.create_state_from_gsd(filename='init.gsd')
sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)

sim.run(1e5)
```

## Change log

[CHANGELOG.rst](CHANGELOG.rst) contains the full change log.

## Contributing to HOOMD-blue

Contributions are welcomed via [pull requests](https://github.com/glotzerlab/hoomd-blue/pulls).
Please report bugs and suggest feature enhancements via the [issue
tracker](https://github.com/glotzerlab/hoomd-blue/issues). See [CONTRIBUTING.rst](CONTRIBUTING.rst)
and [ARCHITECTURE.md](ARCHITECTURE.md) for more information.

## License

**HOOMD-blue** is available under the [3-clause BSD license](LICENSE).
