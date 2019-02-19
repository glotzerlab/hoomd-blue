# HOOMD-blue

[![Citing HOOMD](https://img.shields.io/badge/cite-hoomd-blue.svg)](http://glotzerlab.engin.umich.edu/hoomd-blue/citing.html)
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/hoomd.svg?style=flat)](https://anaconda.org/conda-forge/hoomd)
![Conda](https://img.shields.io/conda/pn/conda-forge/hoomd.svg?colorB=blue&style=flat)
[![Conda](https://img.shields.io/conda/dn/conda-forge/hoomd.svg?style=flat)](https://anaconda.org/conda-forge/hoomd)
[![CircleCI (all branches)](https://img.shields.io/circleci/project/github/glotzerlab/hoomd-blue.svg?style=flat)](https://circleci.com/gh/glotzerlab/hoomd-blue)
[![ReadTheDocs](https://readthedocs.org/projects/hoomd-blue/badge/?version=stable)](https://hoomd-blue.readthedocs.io/en/stable/?badge=stable)
[![Contributors](https://img.shields.io/github/contributors/glotzerlab/hoomd-blue.svg?style=flat)](https://hoomd-blue.readthedocs.io/en/stable/credits.html)
[![hoomd-examples](https://img.shields.io/badge/nbviewer-examples-blueviolet.svg)](https://nbviewer.jupyter.org/github/glotzerlab/hoomd-examples/blob/master/index.ipynb)
[![Docker Image](https://img.shields.io/badge/docker-glotzerlab/software-blue.svg)](https://hub.docker.com/r/glotzerlab/software)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/glotzerlab/hoomd-blue.svg?style=social)](https://github.com/glotzerlab/hoomd-blue)

HOOMD-blue is a general purpose particle simulation toolkit. It performs hard particle Monte Carlo simulations
of a variety of shape classes, and molecular dynamics simulations of particles with a range of pair, bond, angle,
and other potentials. HOOMD-blue runs fast on NVIDIA GPUs, and can scale across
many nodes. For more information, see the [HOOMD-blue website](http://glotzerlab.engin.umich.edu/hoomd-blue).

# Tutorial

[Read the HOOMD-blue tutorial online](http://nbviewer.jupyter.org/github/glotzerlab/hoomd-examples/blob/master/index.ipynb).

## Installing HOOMD-blue

**HOOMD-blue** binary images are available via the [Docker image glotzerlab/software](https://hub.docker.com/r/glotzerlab/software) and the [hoomd package on conda-forge](https://anaconda.org/conda-forge/hoomd). Details on using these images are below.

### Docker images

Pull the [glotzerlab/software](https://hub.docker.com/r/glotzerlab/software/) to get **HOOMD-blue** along with
many other tools commonly used in simulation and analysis workflows. Use these images to execute HOOMD-blue in
Docker/Singularity containers on Mac, Linux, and cloud systems you control and on HPC clusters with Singularity support.
CUDA and MPI operate with native performance on supported HPC systems
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

### Anaconda package

**HOOMD-blue** is available on [conda-forge](https://conda-forge.org/).
To install, first download and install [miniconda](http://conda.pydata.org/miniconda.html).
Then add the `conda-forge` channel and install ``hoomd``:

```bash
$ conda config --add channels conda-forge
$ conda install hoomd
```

If you have already installed ``hoomd`` in ``conda``, you can upgrade to the latest version with:

```bash
$ conda update hoomd
```

### Compile from source

Download source releases directly from the web: https://glotzerlab.engin.umich.edu/Downloads/hoomd

```bash
$ curl -O https://glotzerlab.engin.umich.edu/Downloads/hoomd/hoomd-v2.5.0.tar.gz
```

Or, clone using git:

```bash
$ git clone --recursive https://github.com/glotzerlab/hoomd-blue
```

**HOOMD-blue** uses git submodules. Either clone with the ``--recursive`` option, or execute ``git submodule update --init``
to fetch the submodules.

### Prerequisites

 * Required:
     * Python >= 2.7
     * numpy >= 1.7
     * CMake >= 2.8.0
     * C++ 11 capable compiler (tested with gcc 4.8, 4.9, 5.4, 6.4, 7.0, 8.0, clang 5.0, 6.0)
 * Optional:
     * NVIDIA CUDA Toolkit >= 8.0
     * Intel Threaded Building Blocks >= 4.3
     * MPI (tested with OpenMPI, MVAPICH)
     * LLVM >= 3.6, <= 7.0.0

### Compile

Configure with **cmake** and compile with **make**. Replace ``${PREFIX}`` your desired installation location.

```bash
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=${PREFIX}/lib/python
make install -j10
```

Add ``${PREFIX}/lib/python`` to your ``PYTHONPATH`` to use **HOOMD-blue**.

```bash
$ export PYTHONPATH=$PYTHONPATH:${PREFIX}/lib/python
```

For more detailed instructions, [see the documentation](http://hoomd-blue.readthedocs.io/en/stable/compiling.html).

## Job scripts

HOOMD-blue job scripts are Python scripts. You can control system initialization, run protocols, analyze simulation data,
or develop complex workflows all with Python code in your job.

Here is a simple example.

```python
import hoomd
from hoomd import md
hoomd.context.initialize()

# Create a 10x10x10 square lattice of particles with name A
hoomd.init.create_lattice(unitcell=hoomd.lattice.sc(a=2.0, type_name='A'), n=10)

# Specify Lennard-Jones interactions between particle pairs
nl = md.nlist.cell()
lj = md.pair.lj(r_cut=3.0, nlist=nl)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

# Integrate at constant temperature
all = hoomd.group.all();
md.integrate.mode_standard(dt=0.005)
hoomd.md.integrate.langevin(group=all, kT=1.2, seed=4)

# Run 10,000 time steps
hoomd.run(10e3)
```

Save this as `lj.py` and run with `python lj.py` (or `singularity exec software.simg python3 lj.py` if using containers).

## Reference Documentation

Read the [reference documentation on readthedocs](http://hoomd-blue.readthedocs.io).

## Change log

See [ChangeLog.md](ChangeLog.md).

## Contributing to HOOMD-blue.

See [CONTRIBUTING.md](CONTRIBUTING.md).
