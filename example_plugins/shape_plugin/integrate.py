import hoomd
from hoomd import hpmc
from . import _shape_plugin


class MySphere(hpmc.integrate.HPMCIntegrator)
    """Example shape integrator."""

    # set static class data
    _ext_module = _shape_plugin
    _cpp_cls = "IntegratorHPMCMonoMySphere"

    def __init__(self, default_d=0.1, default_a=0.1, translation_move_probability=0.5, nselect=4):
        # initialize base class
        super().__init__(default_d, default_a, translation_move_probability, nselect)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            radius=float,
                                            ignore_statistics=False,
                                            orientable=False,
                                            len_keys=1))
        self._add_typeparam(typeparam_shape)

    @log(category='object', requires_run=True)
    def type_shapes(self):
        """list[dict]: Description of shapes in ``type_shapes`` format."""
        return super()._return_type_shapes()
