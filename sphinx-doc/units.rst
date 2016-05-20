.. _page-units:

Units
=====

HOOMD-blue stores and computes all values in a system of generic, fully self-consistent set of units.
No conversion factors need to be applied to values at every step. For example, a value with units
of force comes from dividing energy by distance.

Fundamental Units
-----------------

The three fundamental units are:

- distance - :math:`\mathcal{D}`
- energy - :math:`\mathcal{E}`
- mass - :math:`\mathcal{M}`

All other units that appear in HOOMD-blue are derived from these. Values can be converted into any other system
of units by assigning the desired units to :math:`\mathcal{D}`, :math:`\mathcal{E}`, and :math:`\mathcal{M}` and then
multiplying by the appropriate conversion factors.

The standard *Lennard-Jones* symbols :math:`\sigma` and :math:`\epsilon` are intentionally not referred to here.
When you assign a value to :math:`\epsilon` in hoomd, for example, you are assigning it in units of energy:
:math:`\epsilon = 5 \mathcal{E}`. :math:`\epsilon` is **NOT** the unit of energy - it is a value with units of
energy.

Temperature (thermal energy)
----------------------------

HOOMD-blue accepts all temperature inputs and provides all temperature output values in units of energy:
:math:`k T`, where :math:`k` is Boltzmann's constant. When using physical units, the value :math:`k_\mathrm{B}`
is determined by the choices for distance, energy, and mass. In reduced units, one usually reports the value
:math:`T^* = \frac{k T}{\mathcal{E}}`.

Most of the argument inputs in HOOMD take the argument name ``kT`` to make it explicit. A few areas of the code
may still refer to this as ``temperature``.

Charge
------

The unit of charge used in HOOMD-blue is also reduced, but is not represented using just the 3 fundamental units -
the permittivity of free space :math:`\varepsilon_0` is also present. The units of charge are:
:math:`(4 \pi \varepsilon_0 \mathcal{D} \mathcal{E})^{1/2}`. Divide a given charge by this quantity to convert it into
an input value for HOOMD-blue.

Common derived units
--------------------

Here are some commonly used derived units:

- time - :math:`\tau = \sqrt{\frac{\mathcal{M} \mathcal{D}^2}{\mathcal{E}}}`
- volume - :math:`\mathcal{D}^3`
- velocity - :math:`\frac{\mathcal{D}}{\tau}`
- momentum - :math:`\mathcal{M} \frac{\mathcal{D}}{\tau}`
- acceleration - :math:`\frac{\mathcal{D}}{\tau^2}`
- force - :math:`\frac{\mathcal{E}}{\mathcal{D}}`
- pressure - :math:`\frac{\mathcal{E}}{\mathcal{D}^3}`

Example physical units
----------------------

There are many possible choices of physical units that one can assign. One common choice is:

- distance - :math:`\mathcal{D} = \mathrm{nm}`
- energy - :math:`\mathcal{E} = \mathrm{kJ/mol}`
- mass - :math:`\mathcal{M} = \mathrm{amu}`

Derived units / values in this system:

- time - picoseconds
- velocity - nm/picosecond
- k = 0.00831445986144858 kJ/mol/Kelvin
