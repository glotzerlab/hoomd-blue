==========
HOOMD-blue
==========

.. only:: html

    |Citing-HOOMD|
    |conda-forge|
    |conda-forge-Downloads|
    |GitHub Actions|
    |Read-the-Docs|
    |Contributors|
    |License|


    .. |Citing-HOOMD| image:: https://img.shields.io/badge/cite-hoomd-blue.svg
        :target: https://glotzerlab.engin.umich.edu/hoomd-blue/citing.html
    .. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/hoomd.svg?style=flat
        :target: https://anaconda.org/conda-forge/hoomd
    .. |conda-forge-Downloads| image:: https://img.shields.io/conda/dn/conda-forge/hoomd.svg?style=flat
        :target: https://anaconda.org/conda-forge/hoomd
    .. |GitHub Actions| image:: https://github.com/glotzerlab/hoomd-blue/actions/workflows/test.yml/badge.svg
        :target: https://github.com/glotzerlab/hoomd-blue/actions/workflows/test.yml
    .. |Read-the-Docs| image:: https://img.shields.io/readthedocs/hoomd-blue/stable.svg
        :target: https://hoomd-blue.readthedocs.io/en/stable/?badge=stable
    .. |Contributors| image:: https://img.shields.io/github/contributors-anon/glotzerlab/hoomd-blue.svg?style=flat
        :target: https://hoomd-blue.readthedocs.io/en/stable/credits.html
    .. |License| image:: https://img.shields.io/badge/license-BSD--3--Clause-green.svg
        :target: https://github.com/glotzerlab/hoomd-blue/blob/maint/LICENSE

**HOOMD-blue** is a general purpose particle simulation toolkit. It performs
hard particle Monte Carlo simulations of a variety of shape classes, and
molecular dynamics simulations of particles with a range of pair, bond, angle,
and other potentials. **HOOMD-blue** runs fast on NVIDIA GPUs, and can scale
across thousands of nodes. For more information, see the `HOOMD-blue website
<https://glotzerlab.engin.umich.edu/hoomd-blue/>`_.

Resources
=========

- `GitHub Repository <https://github.com/glotzerlab/hoomd-blue>`_:
  Source code and issue tracker.
- :doc:`Installation Guide </installation>`:
  Instructions for installing and compiling **HOOMD-blue**.
- `hoomd-users Google Group <https://groups.google.com/d/forum/hoomd-users>`_:
  Ask questions to the **HOOMD-blue** community.
- `HOOMD-blue website <https://glotzerlab.engin.umich.edu/hoomd-blue/>`_:
  Additional information and publications.

Job scripts
===========

HOOMD-blue job scripts are Python scripts. You can control system
initialization, run protocols, analyze simulation data, or develop complex
workflows all with Python code in your job.

.. toctree::
    :maxdepth: 1
    :caption: Getting started

    installation
    building
    migrating
    changelog
    citing

.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    tutorial/00-Introducing-HOOMD-blue/00-index
    tutorial/01-Introducing-Molecular-Dynamics/00-index
    tutorial/02-Logging/00-index
    tutorial/03-Custom-Actions-In-Python/00-index

.. toctree::
   :maxdepth: 3
   :caption: Stable Python packages

   package-hoomd
   package-hpmc
   package-md

.. toctree::
    :maxdepth: 1
    :caption: Developer guide

    contributing
    style
    testing
    components

.. toctree::
   :maxdepth: 1
   :caption: Reference

   units
   deprecated
   license
   credits
   indices
