# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest

import hoomd
from hoomd.conftest import (Options, Either, Generator, ClassDefinition,
                            pickling_check, logging_check,
                            autotuned_kernel_parameter_check)
from hoomd.logging import LoggerCategories


class MethodDefinition(ClassDefinition):

    def __init__(self,
                 cls,
                 constructor_spec,
                 attribute_spec=None,
                 generator=None,
                 rattle_cls=None,
                 requires_thermostat=False):
        super().__init__(cls, constructor_spec, attribute_spec, generator)
        self.rattle_cls = rattle_cls
        self.requires_thermostat = requires_thermostat

        def default_thermostat():
            return hoomd.md.methods.thermostats.Bussi(2.0)

        self.default_thermostat = default_thermostat
        self.default_filter = hoomd.filter.All()

    def generate_init_args(self):
        kwargs = super().generate_init_args()
        kwargs["filter"] = self.default_filter
        if self.requires_thermostat:
            kwargs["thermostat"] = self.default_thermostat()
        return kwargs

    def generate_all_attr_change(self):
        change_attrs = super().generate_all_attr_change()
        change_attrs["filter"] = hoomd.filter.Type(["A"])
        if self.requires_thermostat:
            change_attrs["thermostat"] = hoomd.md.methods.thermostats.MTTK(
                1.5, 0.05)
        return change_attrs


generator = Generator(np.random.default_rng(546162595))

_method_definitions = (
    MethodDefinition(hoomd.md.methods.ConstantVolume, {}, {},
                     generator,
                     requires_thermostat=True),
    MethodDefinition(hoomd.md.methods.ConstantPressure, {
        "S": Either(hoomd.variant.Variant, (hoomd.variant.Variant,) * 6),
        "tauS": float,
        "couple": Options("xy", "xz", "yz", "xyz"),
        "box_dof": (bool,) * 6,
        "rescale_all": bool,
        "gamma": float
    },
                     generator=generator,
                     requires_thermostat=True),
    MethodDefinition(hoomd.md.methods.DisplacementCapped,
                     {"maximum_displacement": hoomd.variant.Variant},
                     generator=generator),
    MethodDefinition(hoomd.md.methods.Langevin, {
        "kT": hoomd.variant.Variant,
        "alpha": Either(float, None),
        "tally_reservoir_energy": bool
    },
                     generator=generator,
                     rattle_cls=hoomd.md.methods.rattle.Langevin),
    MethodDefinition(hoomd.md.methods.Brownian, {
        "kT": hoomd.variant.Variant,
        "alpha": Either(float, None)
    },
                     generator=generator,
                     rattle_cls=hoomd.md.methods.rattle.Brownian),
    MethodDefinition(hoomd.md.methods.OverdampedViscous,
                     {"alpha": Either(float, None)},
                     generator=generator,
                     rattle_cls=hoomd.md.methods.rattle.OverdampedViscous),
)


@pytest.fixture(scope="module",
                params=_method_definitions,
                ids=lambda x: x.cls.__name__)
def method_definition(request):
    return request.param


_thermostat_definition = (
    # Somewhat hacky way of representing None or no thermostat
    ClassDefinition(lambda: None, {}, generator=generator),
    ClassDefinition(hoomd.md.methods.thermostats.MTTK, {
        "kT": hoomd.variant.Variant,
        "tau": float
    },
                    generator=generator),
    ClassDefinition(hoomd.md.methods.thermostats.Bussi,
                    {"kT": hoomd.variant.Variant},
                    generator=generator),
    ClassDefinition(hoomd.md.methods.thermostats.Berendsen, {
        "kT": hoomd.variant.Variant,
        "tau": float
    },
                    generator=generator),
)


@pytest.fixture(scope="module",
                params=_thermostat_definition,
                ids=lambda x: x.cls.__name__)
def thermostat_definition(request):
    return request.param


def check_instance_attrs(instance, attr_dict, set_attrs=False):

    def equality(a, b):
        if isinstance(a, type(b)):
            return a == b
        if isinstance(a, hoomd.variant.Constant):
            return a.value == b
        if isinstance(b, hoomd.variant.Constant):
            return b.value == a
        return a == b

    for attr, value in attr_dict.items():
        if set_attrs:
            setattr(instance, attr, value)
        instance_value = getattr(instance, attr)
        if hasattr(value, "__iter__") and not isinstance(value, str):
            assert all(equality(a, b) for a, b in zip(value, instance_value))
        elif attr == "S":
            if hasattr(instance_value, "__iter__"):
                assert all(equality(value, a) for a in instance_value[:3])
        elif attr in {"thermostat", "filter"}:
            assert instance_value is value
        else:
            assert equality(instance_value, value)


class TestThermostats:

    @pytest.mark.parametrize("n", range(10))
    def test_attributes(self, thermostat_definition, n):
        """Test the construction and setting of attributes.

        Purposely manually tests the filter and thermostat attributes.
        """
        constructor_args = thermostat_definition.generate_init_args()
        thermostat = thermostat_definition.cls(**constructor_args)
        check_instance_attrs(thermostat, constructor_args)
        check_instance_attrs(thermostat,
                             thermostat_definition.generate_all_attr_change(),
                             True)

    @pytest.mark.parametrize("n", range(10))
    def test_attributes_attached(self, simulation_factory,
                                 two_particle_snapshot_factory,
                                 thermostat_definition, n):
        """Test the setting of attributes with attaching."""
        constructor_args = thermostat_definition.generate_init_args()
        thermostat = thermostat_definition.cls(**constructor_args)

        method = hoomd.md.methods.ConstantVolume(hoomd.filter.All(), thermostat)
        sim = simulation_factory(two_particle_snapshot_factory())
        sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[method])
        sim.run(0)
        check_instance_attrs(thermostat, constructor_args)

        change_attrs = thermostat_definition.generate_all_attr_change()
        if isinstance(thermostat, hoomd.md.methods.thermostats.Berendsen):
            with pytest.raises(hoomd.error.MutabilityError):
                thermostat.tau = change_attrs.pop("tau")
        check_instance_attrs(thermostat, change_attrs, True)

    def test_thermostat_thermalize_thermostat_dof(
            self, simulation_factory, two_particle_snapshot_factory):
        """Tests that NVT.thermalize_thermostat_dof can be called."""
        thermostat = hoomd.md.methods.thermostats.MTTK(1.5, 0.05)
        nvt = hoomd.md.methods.ConstantVolume(thermostat=thermostat,
                                              filter=hoomd.filter.All())

        sim = simulation_factory(two_particle_snapshot_factory())
        sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[nvt])
        sim.run(0)

        thermostat.thermalize_thermostat_dof()
        xi, eta = thermostat.translational_thermostat_dof
        assert xi != 0.0
        assert eta == 0.0

        xi_rot, eta_rot = thermostat.rotational_thermostat_dof
        assert xi_rot == 0.0
        assert eta_rot == 0.0

        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            snap.particles.moment_inertia[:] = [[1, 1, 1], [2, 0, 0]]
        sim.state.set_snapshot(snap)

        xi_rot, eta_rot = thermostat.rotational_thermostat_dof
        assert xi_rot == 0.0
        assert eta_rot == 0.0

    def test_logging(self):
        logging_check(
            hoomd.md.methods.thermostats.MTTK, ('md', 'methods', 'thermostats'),
            {
                'thermostat_energy': {
                    'category': LoggerCategories.scalar,
                    'default': True
                },
            })

    def test_pickling(self, thermostat_definition, simulation_factory,
                      two_particle_snapshot_factory):
        constructor_args = thermostat_definition.generate_init_args()
        thermostat = thermostat_definition.cls(**constructor_args)
        pickling_check(thermostat)

        sim = simulation_factory(two_particle_snapshot_factory())
        method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All(),
                                                 thermostat=thermostat)
        integrator = hoomd.md.Integrator(0.05, methods=[method])
        sim.operations.integrator = integrator
        sim.run(0)
        pickling_check(thermostat)


class TestMethods:

    @pytest.mark.parametrize("n", range(10))
    def test_attributes(self, method_definition, n):
        """Test the construction and setting of attributes."""
        constructor_args = method_definition.generate_init_args()
        method = method_definition.cls(**constructor_args)
        check_instance_attrs(method, constructor_args)
        check_instance_attrs(method,
                             method_definition.generate_all_attr_change(), True)

    @pytest.mark.parametrize("n", range(10))
    def test_attributes_attached(self, simulation_factory,
                                 two_particle_snapshot_factory,
                                 method_definition, n):
        """Test the setting of attributes with attaching."""
        constructor_args = method_definition.generate_init_args()
        method = method_definition.cls(**constructor_args)

        sim = simulation_factory(two_particle_snapshot_factory())
        sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[method])
        sim.run(0)
        check_instance_attrs(method, constructor_args)

        change_attrs = method_definition.generate_all_attr_change()

        # filter cannot be set after scheduling
        with pytest.raises(hoomd.error.MutabilityError):
            method.filter = change_attrs.pop("filter")

        if isinstance(method, hoomd.md.methods.ConstantPressure):
            with pytest.raises(hoomd.error.MutabilityError):
                method.gamma = change_attrs.pop("gamma")

        check_instance_attrs(method, change_attrs, True)

    def test_switch_methods(self, simulation_factory,
                            two_particle_snapshot_factory):
        all_ = hoomd.filter.All()
        method = hoomd.md.methods.Langevin(all_, 1.5, 0.1)

        sim = simulation_factory(two_particle_snapshot_factory())
        sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[method])
        assert len(sim.operations.integrator.methods) == 1
        sim.run(5)
        assert len(sim.operations.integrator.methods) == 1

        sim.operations.integrator.methods.remove(method)
        assert len(sim.operations.integrator.methods) == 0

        sim.operations.integrator.methods.append(
            hoomd.md.methods.ConstantVolume(all_, None))
        assert len(sim.operations.integrator.methods) == 1

    def test_constant_pressure_thermalize_barostat_dof(
            self, simulation_factory, two_particle_snapshot_factory):
        """Tests that ConstantPressure.thermalize_barostat_dof can be called."""
        all_ = hoomd.filter.All()
        npt = hoomd.md.methods.ConstantPressure(
            filter=all_,
            thermostat=hoomd.md.methods.thermostats.Bussi(1.5),
            S=[1, 2, 3, 0.125, 0.25, 0.5],
            tauS=2.0,
            box_dof=[True, True, True, True, True, True],
            couple='xyz')

        sim = simulation_factory(two_particle_snapshot_factory())
        sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[npt])
        sim.run(0)

        npt.thermalize_barostat_dof()
        for v in npt.barostat_dof:
            assert v != 0.0

    def test_constant_pressure_attributes_attached_2d(
            self, simulation_factory, two_particle_snapshot_factory):
        """Test attributes of ConstantPressure specific to 2D simulations."""
        all_ = hoomd.filter.All()
        npt = hoomd.md.methods.ConstantPressure(
            filter=all_,
            thermostat=hoomd.md.methods.thermostats.Bussi(1.0),
            S=2.0,
            tauS=2.0,
            couple='xy')

        sim = simulation_factory(two_particle_snapshot_factory(dimensions=2))
        sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[npt])
        sim.run(0)

        for invalid_couple in ("xyz", "xz", "yz"):
            with pytest.raises(ValueError):
                npt.couple = invalid_couple

        npt.couple = 'none'
        assert npt.couple == 'none'

        npt.box_dof = [True, True, True, True, True, True]
        assert npt.box_dof == [True, True, False, True, False, False]

    def test_kernel_parameters(self, method_definition, simulation_factory,
                               two_particle_snapshot_factory):
        sim = simulation_factory(two_particle_snapshot_factory())
        constructor_args = method_definition.generate_init_args()
        method = method_definition.cls(**constructor_args)
        integrator = hoomd.md.Integrator(0.05, methods=[method])
        sim.operations.integrator = integrator
        sim.run(0)
        autotuned_kernel_parameter_check(instance=method,
                                         activate=lambda: sim.run(1))

    def test_pickling(self, method_definition, simulation_factory,
                      two_particle_snapshot_factory):
        constructor_args = method_definition.generate_init_args()
        method = method_definition.cls(**constructor_args)
        pickling_check(method)

        sim = simulation_factory(two_particle_snapshot_factory())
        integrator = hoomd.md.Integrator(0.05, methods=[method])
        sim.operations.integrator = integrator
        sim.run(0)
        pickling_check(method)

    def test_logging(self):
        logging_check(
            hoomd.md.methods.ConstantPressure, ('md', 'methods'), {
                'barostat_energy': {
                    'category': LoggerCategories.scalar,
                    'default': True
                },
            })
        logging_check(
            hoomd.md.methods.thermostats.MTTK, ('md', 'methods', 'thermostats'),
            {
                'thermostat_energy': {
                    'category': LoggerCategories.scalar,
                    'default': True
                },
            })


@pytest.fixture(scope="module", params=range(7))
def manifold(request):
    return (
        hoomd.md.manifold.Cylinder(r=5),
        hoomd.md.manifold.Diamond(N=(1, 1, 1)),
        hoomd.md.manifold.Ellipsoid(a=3.3, b=5, c=4.1),
        hoomd.md.manifold.Gyroid(N=(1, 2, 1)),
        hoomd.md.manifold.Primitive(N=(1, 1, 1)),
        hoomd.md.manifold.Sphere(r=5),
        hoomd.md.manifold.Plane(),
    )[request.param]


class TestRattle:

    @pytest.mark.skip("rattle not yet ported.")
    def test_rattle_attributes(self, method_definition, manifolds):
        if method_definition.rattle_method is None:
            pytest.skip("RATTLE method is not implemented for this method")

        all_ = hoomd.filter.All()
        constructor_args = method_definition.generate_init_args()
        method = method_definition.rattle_method(**constructor_args,
                                                 filter=all_,
                                                 manifold_constraint=manifold)
        assert method.manifold_constraint == manifold
        assert method.tolerance == 1e-6
        check_instance_attrs(method, constructor_args)

        sphere = hoomd.md.manifold.Sphere(r=10)
        with pytest.raises(AttributeError):
            method.manifold_constraint = sphere
        assert method.manifold_constraint == manifold

        method.tolerance = 1e-5
        assert method.tolerance == 1e-5
        check_instance_attrs(method,
                             method_definition.generate_all_attr_change(), True)

    @pytest.mark.skip("rattle not yet ported.")
    def test_rattle_attributes_attached(self, simulation_factory,
                                        two_particle_snapshot_factory,
                                        method_definition, manifold):

        if method_definition.rattle_method is None:
            pytest.skip("RATTLE integrator is not implemented for this method")

        all_ = hoomd.filter.All()
        constructor_args = method_definition.generate_init_args()
        method = method_definition.rattle_method(**constructor_args,
                                                 filter=all_,
                                                 manifold_constraint=manifold)

        sim = simulation_factory(two_particle_snapshot_factory())
        sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[method])
        sim.run(0)

        assert method.filter is all_
        assert method.manifold_constraint == manifold
        assert method.tolerance == 1e-6
        check_instance_attrs(method, constructor_args)

        type_A = hoomd.filter.Type(['A'])
        with pytest.raises(AttributeError):
            # filter cannot be set after scheduling
            method.filter = type_A

        sphere = hoomd.md.manifold.Sphere(r=10)
        with pytest.raises(AttributeError):
            # manifold cannot be set after scheduling
            method.manifold_constraint = sphere
        assert method.manifold_constraint == manifold

        method.tolerance = 1e-5
        assert method.tolerance == 1e-5

        check_instance_attrs(method,
                             method_definition.generate_all_attr_change(), True)

    @pytest.mark.skip("rattle not yet ported.")
    def test_rattle_switch_methods(self, simulation_factory,
                                   two_particle_snapshot_factory):
        if method_definition.rattle_method is None:
            pytest.skip("RATTLE integrator is not implemented for this method")

        sim = simulation_factory(two_particle_snapshot_factory())

        all_ = hoomd.filter.All()
        manifold = hoomd.md.manifold.Sphere(5.0)
        method = hoomd.md.methods.rattle.Langevin(all_, 1.5, manifold)
        sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[method])
        sim.run(0)

        assert len(sim.operations.integrator.methods) == 1
        sim.operations.integrator.methods.remove(method)
        assert len(sim.operations.integrator.methods) == 0

        sim.operations.integrator.methods.append(
            hoomd.md.methods.rattle.NVE(filter=all_,
                                        manifold_constraint=manifold))
        assert len(sim.operations.integrator.methods) == 1
