.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

Units
+++++

HOOMD-blue does not adopt a particular system of units, nor does it offer a variety of systems
to choose from. Instead, it follows a self-consistent system of units where all derived units
(e.g. force) are defined in terms of base units (e.g. energy / length). To adopt a system of units
for your simulations, choose a set of base units (e.g. meters versus centimeters for length), and
then determine what the derived units are.

Base Units
==========

The base units are:

- :math:`[\mathrm{energy}]`
- :math:`[\mathrm{length}]`
- :math:`[\mathrm{mass}]`

Unit Conversion
===============

Example unit conversions between derived units and base units:

.. list-table::
   :header-rows: 1

   * - Derived units
     - Relation to base units
   * - :math:`[\mathrm{area}]`
     - :math:`[\mathrm{length}]^2`
   * - :math:`[\mathrm{volume}]`
     - :math:`[\mathrm{length}]^3`
   * - :math:`[\mathrm{time}]`
     - :math:`[\mathrm{energy}]^{-1/2} \cdot [\mathrm{length}] \cdot [\mathrm{mass}]^{1/2}`
   * - :math:`[\mathrm{velocity}]`
     - :math:`[\mathrm{energy}]^{1/2} \cdot [\mathrm{mass}]^{-1/2}`
   * - :math:`[\mathrm{force}]`
     - :math:`[\mathrm{energy}] \cdot [\mathrm{length}]^{-1}`
   * - :math:`[\mathrm{pressure}]`
     - :math:`[\mathrm{energy}] \cdot [\mathrm{length}]^{-3}`
   * - :math:`[\mathrm{charge}]`
     - :math:`\left(4 \pi \epsilon_{0} \cdot [\mathrm{energy}] \cdot [\mathrm{length}] \right)^{1/2}`
       - where :math:`\epsilon_{0}` is permittivity of free space

.. note::

    Most of the units on this page apply to MD simulations.

    In HPMC, the primary unit is that of length. Mass is factored out of the partition function and
    does not enter into the simulation. In addition, the energy scale is irrelevant in athermal
    HPMC systems where overlapping energies are infinite and valid configurations have
    zero potential energy. However, energy does appear implicitly in derived units like
    :math:`[\mathrm{pressure}] = [\mathrm{energy}] \cdot [\mathrm{length}]^{-3}`.  In
    HPMC, :math:`kT` is set to 1 :math:`\mathrm{energy}`.

Common unit systems
===================

Example base and derived units for common MD unit systems.

.. note::

    All conversion factors given here are computed with Wolfram Alpha using the provided links.

.. list-table::
   :header-rows: 1

   * - Unit
     - AKMA
     - MD
   * - :math:`[\mathrm{energy}]`
     - kcal/mol
     - kJ/mol
   * - :math:`[\mathrm{length}]`
     - Å
     - nm
   * - :math:`[\mathrm{mass}]`
     - atomic mass unit
     - atomic mass unit
   * - :math:`[\mathrm{area}]`
     - :math:`\mathrm{Å}^2`
     - :math:`\mathrm{nm}^2`
   * - :math:`[\mathrm{volume}]`
     - :math:`\mathrm{Å}^3`
     - :math:`\mathrm{nm}^3`
   * - :math:`[\mathrm{time}]`
     - `48.8882129 fs <https://www.wolframalpha.com/input/?i=angstrom+*+amu%5E%281%2F2%29+*+%28kcal%2FAvogadro+number%29%5E%28%E2%88%921%2F2%29>`__
     - `1 ps <https://www.wolframalpha.com/input/?i=nanometer+*+amu%5E%281%2F2%29+*+%28kilojoule%2FAvogadro+number%29%5E%28%E2%88%921%2F2%29>`__
   * - :math:`[\mathrm{velocity}]`
     - `0.02045482828 Å/fs <https://www.wolframalpha.com/input/?i=%28kcal%2FAvogadro+number%29%5E%281%2F2%29+*+amu%5E%28-1%2F2%29+in+angstrom%2Ffs>`__
     - 1 nm/ps
   * - :math:`[\mathrm{force}]`
     - kcal/mol/Å
     - kJ/mol/nm
   * - :math:`[\mathrm{pressure}]`
     - `68568.4230 atm <https://www.wolframalpha.com/input/?i=%28kcal%2FAvogadro+number%29+*+angstrom%5E%28-3%29+in+atmospheres>`__
     - `16.3882464 atm <https://www.wolframalpha.com/input/?i=%28kilojoule%2FAvogadro+number%29+*+nanometer%5E%28-3%29+in+atmospheres>`__
   * - :math:`[\mathrm{charge}]`
     - `0.05487686461 e <https://www.wolframalpha.com/input/?i=sqrt%284+*+pi+*+permittivity+of+free+space+*+1+%28kcal%2FAvogadro%27s+number%29+*+1+angstrom%29+%2F+proton+charge>`__
     - `0.0848385920 e <https://www.wolframalpha.com/input/?i=sqrt%284+*+pi+*+permittivity+of+free+space+*+1+%28kilojoule%2FAvogadro%27s+number%29+*+1+nanometer%29+%2F+proton+charge>`__
   * - :math:`k` (Boltzmann's constant)
     - `0.00198720426 kcal/mol/K <https://www.wolframalpha.com/input/?i=boltzmann%27s+constant+in+kcal%2FAvogadro+number%2FK>`__
     - `0.00831446262 kJ/mol/K <https://www.wolframalpha.com/input/?i=boltzmann%27s+constant+in+kilojoues%2FAvogadro+number%2FK>`__
