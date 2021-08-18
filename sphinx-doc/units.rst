Units
+++++

HOOMD-blue does not adopt any particular system of units, but instead follows a self-consistent system of units. Users are free to define HOOMD-blue's base units (e.g. meters versus centimeters for length), and units of all derived quantities can be determined from the base units. For instance if the base units are chosen to be 1 meter, 1 Joule, and 1 kilogram, then velocity is in meters per second, but if one chooses nanometers for length, kilojoules per mole for energy, and 1 amu for mass then velocity is in nanometers per picosecond.

Note:
    In HPMC, the primary unit is that of length. Mass is factored out of the partition function and
    does not enter into the simulation. In addition, the scale of energy is irrelevant in athermal
    HPMC systems where overlapping energies are infinite and valid configurations have
    zero potential energy. However, energy does appear implicitly in derived units like
    :math:`[\mathrm{pressure}] = \left(\frac{\mathrm{[energy]}}{\mathrm{[length]}^3}\right)`.  In
    HPMC, :math:`kT` is assumed to be 1 unit of energy.

Base Units
==========

Base units in HOOMD-blue are:

- energy
- length
- mass

Unit Conversion
===============

Unit conversions between derived units and base units:


.. list-table::
   :header-rows: 1

   * - Derived units
     - Base units
   * - :math:`[\mathrm{time}]`
     - :math:`[\sqrt{\mathrm{mass} \cdot \mathrm{length}^2 \cdot \mathrm{energy}^{-1}}]`
   * - :math:`[\mathrm{force}]`
     - :math:`[\mathrm{energy} \cdot \mathrm{length}^{-1}]`
   * - :math:`[\mathrm{pressure}]`
     - :math:`[\mathrm{energy} \cdot \mathrm{length}^{-3}]`
   * - :math:`[\mathrm{charge}]`
     - :math:`[\sqrt{4 \cdot \pi \cdot \epsilon_{0} \cdot \mathrm{energy} \cdot \mathrm{length}}]`
       :math:`\epsilon_{0}`: the permittivity of free space
   * - :math:`[\mathrm{velocity}]`
     - :math:`[\sqrt{\mathrm{energy} \cdot \mathrm{mass}^{-1}}]`
   * - :math:`[\mathrm{volume}]`
     - :math:`[\mathrm{length}^3]`
   * - :math:`[\mathrm{area}]`
     - :math:`[\mathrm{length}^2]`
