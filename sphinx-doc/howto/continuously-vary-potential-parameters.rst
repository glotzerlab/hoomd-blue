.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

How to continuously vary potential parameters
=============================================

To continuously vary potential parameters (and other non-variant parameters) during a HOOMD-blue
simulation:

1. Write a class that subclasses `hoomd.custom.Action` and implements the desired change as a
   function of timestep in ``act``.
2. Create a `hoomd.update.CustomUpdater` and pass an instance of your action.
3. Add the updater to the simulation operations.

For example:

.. literalinclude:: continuously-vary-potential-parameters.py
    :language: python
