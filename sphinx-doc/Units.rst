Unit Conversion
===================

Unit conversion between dervied units and based units

.. list-table::
   :header-rows: 1

   * - Feature
     - Replace with
   * - Python 2.7
     - Python >= 3.6
   * - ``static`` parameter in ``hoomd.dump.gsd``
     - ``dynamic`` parameter
   * - ``set_params`` and other ``set_*`` methods
     - Parameters and type parameters accessed by properties.
   * - ``context.initialize``
     - ``device.CPU`` / ``device.GPU``
   * - ``util.quiet_status`` and ``util.unquiet_status``
     - No longer needed.

