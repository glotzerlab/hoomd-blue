# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement variants that return box parameters as a function of time."""

from hoomd import _hoomd, Box
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import box_preprocessing, variant_preprocessing


class BoxVariant(_hoomd.VectorVariantBox):
    """Box-like vector variant base class.

    `hoomd.variant.box.BoxVariant` provides an interface to length-6 vector variants
    that are valid `hoomd.box.box_like` objects.  The return value of the
    ``__call__`` method returns a length-6 array of scalar values that represent
    the quantities ``Lx``, ``Ly``, ``Lz``, ``xy``, ``xz``, and ``yz`` of a
    simulation box.
    """
    pass


class Constant(_hoomd.VectorVariantBoxConstant, BoxVariant):
    """A constant box variant.

    Args:
        box (hoomd.box.box_like): The box.

    `Constant` returns ``[box.Lx, box.Ly, box.Lz, box.xz, box.xz, box.yz]`` at
    all time steps.

    """

    def __init__(self, box):
        box = box_preprocessing(box)
        BoxVariant.__init__(self)
        _hoomd.VectorVariantBoxConstant.__init__(self, box._cpp_obj)

    @property
    def box(self):
        """hoomd.Box: The box."""
        return Box._from_cpp(self._box)

    @box.setter
    def box(self, box):
        box = box_preprocessing(box)
        self._box = box._cpp_obj



class Interpolate(_hoomd.VectorVariantBoxInterpolate, BoxVariant):
    """Interpolate between two boxes linearly in time.

    Args:
        initial_box (hoomd.box.box_like): The initial box.
        final_box (hoomd.box.box_like): The final box.
        variant (hoomd.variant.variant_like): A variant used to interpolate
            between the two boxes.

    Ramp returns the array corresponding to *initial_box* for
    :math:`t \\leq t_{\\mathrm{start}}` and *final_box* for
    :math:`t \\geq t_{\\mathrm{start}} + t_{\\mathrm{ramp}}`.

    Attributes:
        variant (hoomd.variant.Variant): A variant used to interpolate between
            the two boxes.
    """

    def __init__(self, initial_box, final_box, variant):
        box1 = box_preprocessing(initial_box)
        box2 = box_preprocessing(final_box)
        variant = variant_preprocessing(variant)
        BoxVariant.__init__(self)
        _hoomd.VectorVariantBoxInterpolate.__init__(self, box1._cpp_obj,
                                               box2._cpp_obj, variant)

    @property
    def initial_box(self):
        """hoomd.Box: the initial box."""
        return Box._from_cpp(self._initial_box)

    @initial_box.setter
    def initial_box(self, box):
        box = box_preprocessing(box)
        self._initial_box = box._cpp_obj

    @property
    def final_box(self):
        """hoomd.Box: the final box."""
        return Box._from_cpp(self._final_box)
   
    @final_box.setter
    def final_box(self, box):
        box = box_preprocessing(box)
        self._final_box = box._cpp_obj


class InverseVolumeRamp(_hoomd.VectorVariantBoxInverseVolumeRamp, BoxVariant):
    """Produce box arrays whose inverse volume changes linearly with time.

    Args:
        initial_box (hoomd.box.box_like): The initial box.
        final_volume (float): The final volume of the box.
        t_start (int): The time step at the start of the ramp.
        t_ramp (int): The length of the ramp.

    ``InverseVolumeRamp`` produces box arrays that correspond to a box whose
    **inverse volume** (i.e., density for a constant number of particles) varies
    linearly with time. The shape of the box remains constant, that is, the
    ratios of the lengths of the box vectors (:math:`L_y / L_x` and
    :math:`L_z / L_x` and the tilt factors (:math:`xy`, :math:`xz`, :math:`yz`)
    remain constant.

    Attributes:
        final_volume (float): The volume of the final box.
        t_start (int): The time step at the start of the ramp.
        t_ramp (int): The length of the ramp.
    """

    def __init__(self, initial_box, final_volume, t_start, t_ramp):
        BoxVariant.__init__(self)
        box = box_preprocessing(initial_box)
        _hoomd.VectorVariantBoxInverseVolumeRamp.__init__(
            self, box._cpp_obj, final_volume, t_start, t_ramp)

    @property
    def initial_box(self):
        """hoomd.Box: the initial box."""
        return Box._from_cpp(self._initial_box)
    
    @initial_box.setter
    def initial_box(self, box):
        box = box_preprocessing(box)
        self._initial_box = box._cpp_obj
