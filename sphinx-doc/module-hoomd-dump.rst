hoomd.dump
----------

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    hoomd.dump.dcd
    hoomd.dump.getar
    hoomd.dump.gsd

.. rubric:: Details

.. automodule:: hoomd.dump
    :synopsis: Write system configurations to files.
    :exclude-members: dcd, getar, gsd

    .. autoclass:: dcd

    .. autoclass:: getar
        :exclude-members: set_period, DumpProp

        .. class:: DumpProp(name, highPrecision=False, compression=hoomd.dump.getar.Compression.FastCompress)

            Create a dump property specification.

            :param name: Name of the property to dump
            :param highPrecision: True if the property should be dumped in high precision, if possible
            :param compression: Compression level to save the property with, if possible

        .. automethod:: getar.__init__

    .. autoclass:: gsd
