# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Orientation-dependent external field for boundary regions."""

from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import positive_real, nonnegative_real, NDArrayValidator
from hoomd.hpmc import _hpmc
import hoomd
import numpy as np
import inspect

from .external import External


@hoomd.logging.modify_namespace(("hpmc", "external", "OrientationField"))
class OrientationField(External):
    r"""Constrain particle orientations in boundary regions.

    Args:
        kappa (float): Alignment strength in units of kT (default: 100.0).
            :math:`[\mathrm{dimensionless}]`.
        target_orientation (list[float]): Target quaternion orientation as
            [w, x, y, z]. Default is [1, 0, 0, 0] for z-axis alignment.
            :math:`[\mathrm{dimensionless}]`.
        symmetries (numpy.ndarray): Symmetrically equivalent orientations due to
            particle shape symmetry, as an (N_sym, 4) array of quaternions.
            Default is identity quaternion [1, 0, 0, 0] (no additional symmetries).
            For ellipsoids, use [[1,0,0,0], [0,0,0,1]] to account for +z/-z symmetry.
            :math:`[\mathrm{dimensionless}]`.
        field_axis (int): Axis along which to apply the field (0=x, 1=y, 2=z).
            Default is 0 (x-axis). :math:`[\mathrm{dimensionless}]`.
        field_extent (float): Fraction of box that constitutes the boundary regions
            on each end. Must be between 0.0 and 0.5. Default is 1/6 (so boundary
            regions are the outer 1/6 on each side, leaving middle 2/3 free).
            :math:`[\mathrm{dimensionless}]`.

    `OrientationField` applies an energy penalty to particles that are
    misaligned with a target orientation when located in boundary regions
    of the simulation box. This is designed for liquid crystal bend deformation
    simulations following the de Pablo methodology (Soft Matter, 2014).

    The energy function is:

    .. math::

        U_{\mathrm{external},i} = \begin{cases}
            \kappa \cdot (1 - \max_s |\mathbf{q}_i \cdot (\mathbf{q}_{\mathrm{target}} \cdot \mathbf{q}_s)|)^2
            & \text{if in boundary region} \\
            0 & \text{otherwise}
        \end{cases}

    where:
        - :math:`\kappa` is the alignment strength (``kappa`` parameter)
        - :math:`\mathbf{q}_i` is the particle's quaternion orientation
        - :math:`\mathbf{q}_{\mathrm{target}}` is the target orientation
        - :math:`\mathbf{q}_s` are the symmetry quaternions
        - Boundary regions are defined by ``field_axis`` and ``field_extent``
        - The max is taken over all symmetries to find best alignment

    The boundary regions are at both ends of the specified axis. For example,
    with ``field_axis=0`` (x) and ``field_extent=1/6``, the boundaries are
    at x < -Lx/3 and x > Lx/3.

    Note:
        The quaternion dot product uses absolute value, so both aligned and
        anti-aligned orientations have zero energy.

    {inherited}

    ----------

    **Members defined in** `OrientationField`:

    Attributes:
        kappa (float): The alignment strength :math:`[\mathrm{dimensionless}]`.
        target_orientation (numpy.ndarray): The target quaternion orientation
            :math:`[\mathrm{dimensionless}]`.
        symmetries (numpy.ndarray): Symmetrically equivalent orientations
            :math:`[\mathrm{dimensionless}]`.
        field_axis (int): Axis along which field is applied (0=x, 1=y, 2=z)
            :math:`[\mathrm{dimensionless}]`.
        field_extent (float): Fraction of box that is boundary region on each end
            :math:`[\mathrm{dimensionless}]`.
    """

    __doc__ = inspect.cleandoc(__doc__).replace(
        "{inherited}", inspect.cleandoc(External._doc_inherited)
    )

    def __init__(self, kappa=100.0, target_orientation=None, symmetries=None, 
                 field_axis=0, field_extent=1.0/6.0):
        if target_orientation is None:
            target_orientation = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion (z-aligned)
        
        if symmetries is None:
            # Default: identity quaternion (no additional symmetries beyond built-in +/-)
            symmetries = [[1.0, 0.0, 0.0, 0.0]]
        
        # Validate field_axis
        if not isinstance(field_axis, int) or field_axis < 0 or field_axis > 2:
            raise ValueError("field_axis must be 0 (x), 1 (y), or 2 (z)")
        
        # Validate field_extent
        if field_extent < 0.0 or field_extent > 0.5:
            raise ValueError("field_extent must be between 0.0 and 0.5")
        
        param_dict = ParameterDict(
            kappa=float,
            target_orientation=NDArrayValidator(dtype=np.double, shape=(4,)),
            symmetries=NDArrayValidator(dtype=np.double, shape=(None, 4)),
            field_axis=int,
            field_extent=float,
        )
        param_dict["kappa"] = float(kappa)
        param_dict["target_orientation"] = np.array(target_orientation, dtype=np.double)
        param_dict["symmetries"] = np.array(symmetries, dtype=np.double)
        param_dict["field_axis"] = int(field_axis)
        param_dict["field_extent"] = float(field_extent)
        
        self._param_dict.update(param_dict)

    def _make_cpp_obj(self):
        integrator = self._simulation.operations.integrator
        
        # Build C++ class name from integrator's Python class name
        # e.g., "Ellipsoid" -> "ExternalFieldOrientationEllipsoid"
        cpp_cls_name = "ExternalFieldOrientation" + integrator.__class__.__name__
        
        try:
            cpp_cls = getattr(_hpmc, cpp_cls_name)
        except AttributeError:
            raise RuntimeError(
                f"OrientationField is not supported for integrator type "
                f"{integrator.__class__.__name__}. Expected C++ class: {cpp_cls_name}"
            )
        
        # Create the C++ external field object
        # Pass SystemDefinition, kappa, target_orientation, symmetries, field_axis, field_extent
        cpp_sys_def = self._simulation.state._cpp_sys_def
        return cpp_cls(cpp_sys_def, 
                      float(self.kappa), 
                      self.target_orientation, 
                      self.symmetries,
                      int(self.field_axis),
                      float(self.field_extent))

    @hoomd.logging.log(category="object", requires_run=True)
    def energy(self):
        """float: Total energy from the external field :math:`[\\mathrm{energy}]`."""
        if hasattr(self, "_cpp_obj") and self._attached:
            timestep = self._simulation.timestep
            return self._cpp_obj.totalEnergy(timestep)
        return 0.0
