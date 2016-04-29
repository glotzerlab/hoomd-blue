.. _page-units:

Units
=====

HOOMD-blue stores and computes all values in a system of generic, but fully self-consistent set of units.
No conversion factors need to be applied to values at every step. For example, a value with units
of force comes from dividing energy by distance. You may be familiar with this system of units
as being referred to as reduced units. These will be more formally generalized here for
application to all types of potentials in HOOMD-blue.

Fundamental Units
-----------------

The three fundamental units are:

- distance - :math:`\mathcal{D}`
- energy - :math:`\mathcal{E}`
- mass - :math:`\mathcal{M}`

All other units that appear in HOOMD-blue are derived from these. Values can be converted into any other system
of units by assigning the desired units to :math:`\mathcal{D}`, :math:`\mathcal{E}`, and :math:`\mathcal{M}` and then
multiplying by the appropriate conversion factors.

The standard *Lennard-Jones* symbols :math:`\sigma` and :math:`\epsilon` are intentionally not used in this
document. When you assign a value to :math:`\epsilon` in hoomd, for example, you are assigning it in units of energy:
:math:`\epsilon = 5 \mathcal{E}`. Here, :math:`\epsilon` is **NOT** the unit of energy. To understand this in a trivial case,
consider a system with two particle types, there are three distinct :math:`\epsilon_{ij}` values to set, and they cannot
each be the unit of energy.

Temperature (thermal energy)
----------------------------

The standard nomenclature in the literature regarding *reduced temperature*
is generally not very precise or consistent. HOOMD-blue's parameter names unfortunately do not help that situation
(maybe a later version will fix this). Formally, whenever HOOMD-blue asks for or reports a **temperature** :math:`T`, the
value is a thermal energy :math:`T = k_\mathrm{B} T_\mathrm{actual}` *in units of energy*. The value :math:`k_\mathrm{B}`
is determined by your choice of real units for distance, energy, and mass.

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
