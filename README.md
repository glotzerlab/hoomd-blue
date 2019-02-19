# HOOMD-blue

[![Citing HOOMD](https://img.shields.io/badge/cite-hoomd-blue.svg)](https://glotzerlab.engin.umich.edu/hoomd-blue/citing.html)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/hoomd.svg?style=flat)](https://anaconda.org/conda-forge/hoomd)
[![docker](https://img.shields.io/badge/docker-glotzerlab/software-blue.svg)](https://hub.docker.com/r/glotzerlab/software)
[![conda-forge Downloads](https://img.shields.io/conda/dn/conda-forge/hoomd.svg?style=flat)](https://anaconda.org/conda-forge/hoomd)
[![CircleCI (all branches)](https://img.shields.io/circleci/project/github/glotzerlab/hoomd-blue.svg?style=flat)](https://circleci.com/gh/glotzerlab/hoomd-blue)
[![Read the Docs](https://img.shields.io/readthedocs/hoomd-blue/stable.svg)](https://hoomd-blue.readthedocs.io/en/stable/?badge=stable)
[![Contributors](https://img.shields.io/github/contributors/glotzerlab/hoomd-blue.svg?style=flat)](https://hoomd-blue.readthedocs.io/en/stable/credits.html)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

**HOOMD-blue** is a general purpose particle simulation toolkit.It performs hard particle Monte Carlo simulations
of a variety of shape classes, and molecular dynamics simulations of particles with a range of pair, bond, angle,
and other potentials. **HOOMD-blue** runs fast on NVIDIA GPUs, and can scale across thousands of nodes.
For more information, see the [**HOOMD-blue** website](https://glotzerlab.engin.umich.edu/hoomd-blue/).

## Resources

- [Reference Documentation](https://hoomd-blue.readthedocs.io/):
  Full package Python API, usage information, and feature reference.
- [hoomd-users Google Group](https://groups.google.com/d/forum/hoomd-users):
  Ask questions to the **HOOMD-blue** community.
- [**HOOMD-blue** Tutorial](https://nbviewer.jupyter.org/github/glotzerlab/hoomd-examples/blob/master/index.ipynb):
  Beginner's guide, code examples, and sample scripts.
- [**HOOMD-blue** website](https://glotzerlab.engin.umich.edu/hoomd-blue/):
  Additional information, benchmarks, and publications.

## Installation

**HOOMD-blue** binary images are available via the
[Docker image glotzerlab/software](https://hub.docker.com/r/glotzerlab/software) and for Linux and macOS via the
[hoomd package on conda-forge](https://anaconda.org/conda-forge/hoomd). See below for details on using these images.

### Docker image

Pull the [glotzerlab/software](https://hub.docker.com/r/glotzerlab/software/) image to get **HOOMD-blue** along with
many other tools commonly used in simulation and analysis workflows. Use this image to execute **HOOMD-blue** in
Docker/Singularity containers on macOS, Linux, cloud systems you control, or HPC clusters with Singularity support.
CUDA and MPI operate with native performance on supported HPC systems.
See full usage information on the [glotzerlab/software docker hub page](https://hub.docker.com/r/glotzerlab/software/).

Singularity:
```bash
$ umask 002
$ singularity pull docker://glotzerlab/software
```

Docker:
```bash
$ docker pull glotzerlab/software
```

### Conda package

**HOOMD-blue** is available for Linux and macOS via [conda-forge](https://conda-forge.org/).
To install, first download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html).
Then add the `conda-forge` channel and install ``hoomd``:

```bash
$ conda config --add channels conda-forge
$ conda install hoomd
```

If you have already installed ``hoomd`` with ``conda``, you can upgrade to the latest version with:

```bash
$ conda update hoomd
```

### Compile from source

Download source releases directly from the web: https://glotzerlab.engin.umich.edu/Downloads/hoomd

```bash
$ curl -O https://glotzerlab.engin.umich.edu/Downloads/hoomd/hoomd-v2.5.0.tar.gz
```

Or clone using git:

```bash
$ git clone --recursive https://github.com/glotzerlab/hoomd-blue
```

**HOOMD-blue** uses git submodules.
Clone with the ``--recursive`` option or execute ``git submodule update --init`` to fetch the submodules.

#### Prerequisites

 * Required:
     * Python >= 2.7
     * numpy >= 1.7
     * CMake >= 2.8.0
     * C++11 capable compiler (tested with gcc 4.8, 4.9, 5.4, 6.4, 7.0, 8.0, clang 5.0, 6.0)
 * Optional:
     * NVIDIA CUDA Toolkit >= 8.0
     * Intel Threaded Building Blocks >= 4.3
     * MPI (tested with OpenMPI, MVAPICH)
     * LLVM >= 3.6, <= 7.0.0

#### Compile

Configure with `cmake` and compile with `make`. Replace `${PREFIX}` with your desired installation location.

```bash
$ mkdir build
$ cd build
$ cmake ../ -DCMAKE_INSTALL_PREFIX=${PREFIX}/lib/python
$ make install -j10
```

Add `${PREFIX}/lib/python` to your `PYTHONPATH` to use **HOOMD-blue**.

```bash
$ export PYTHONPATH=$PYTHONPATH:${PREFIX}/lib/python
```

For more detailed instructions, [see the documentation](https://hoomd-blue.readthedocs.io/en/stable/compiling.html).

## Job scripts

HOOMD-blue job scripts are Python scripts. You can control system initialization, run protocols, analyze simulation data,
or develop complex workflows all with Python code in your job.

Here is a simple example:

```python
import hoomd
from hoomd import md
hoomd.context.initialize()

# Create a 10x10x10 square lattice of particles with type name A
hoomd.init.create_lattice(unitcell=hoomd.lattice.sc(a=2.0, type_name='A'), n=10)

# Specify Lennard-Jones interactions between particle pairs
nl = md.nlist.cell()
lj = md.pair.lj(r_cut=3.0, nlist=nl)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

# Integrate at constant temperature
all = hoomd.group.all();
md.integrate.mode_standard(dt=0.005)
hoomd.md.integrate.langevin(group=all, kT=1.2, seed=4)

# Run for 10,000 time steps
hoomd.run(10e3)
```

Save this script as `lj.py` and run it with `python lj.py` (or `singularity exec software.simg python3 lj.py` if using Singularity containers).

## Change log

See [ChangeLog.md](ChangeLog.md).

## Contributing to HOOMD-blue

Contributions are welcomed via [pull requests on GitHub](https://github.com/glotzerlab/hoomd-blue/pulls). Please report bugs and suggest feature enhancements via the [issue tracker](https://github.com/glotzerlab/hoomd-blue/issues). See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.
