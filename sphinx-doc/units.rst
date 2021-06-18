Units
+++++

HOOMD-blue follow self-consistent units. This means that users are free to define the units (e.g. meters versus centimeters for length) of HOOMD-blue's base units, and units of all derived quantities can be determined from the user selected units. For instance if the base length is chosen to be 1 meter, energy 1 Joule, and mass 1 kilogram, then velocity is in meters per second, but if one nanometer is chosen for length, kilojoules per mole for energy, and 1 amu for mass then velocity is in nanometers per picosecond.

Base Units
============

Base units in HOOMD-blue are
- energy
- length
- mass

Unit Conversion
===================

Unit coversions between dervied units and base units.


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
       :math:`\epsilon_{0}`: the permitivitty of free space
   * - :math:`[\mathrm{velocity}]`
     - :math:`[\sqrt{\mathrm{energy} \cdot \mathrm{mass}^{-1}}]`
   * - :math:`[\mathrm{volume}]`
     - :math:`[\mathrm{length}^3]`
   * - :math:`[\mathrm{area}]`
     - :math:`[\mathrm{length}^2]`
