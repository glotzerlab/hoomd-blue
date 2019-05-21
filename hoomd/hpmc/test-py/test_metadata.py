import unittest
import tempfile
import json
import warnings

import hoomd
from hoomd import hpmc

from hoomd_test_metadata import TestMetadata, TestWithFile, unsupported

class TestWithIntegrator(TestMetadata):
    """Ensure that integrators are made when needed."""
    def preprocess(self):
        self.mc_seed = 10809
        self.mc = hpmc.integrate.sphere(self.mc_seed)


class TestAnalyzeSdf(TestWithIntegrator, TestWithFile):
    """Tests hpmc.analyze.sdf."""
    def build_object(self):
        return hpmc.analyze.sdf(
            self.mc, self.file.name, xmax=10, dx=0.1, navg=100, period=10)


class TestComputeFreevolume(TestWithIntegrator, unittest.TestCase):
    """Tests hpmc.compute.free_volume."""
    def build_object(self):
        return hpmc.compute.free_volume(self.mc, 839)


@unsupported
class TestComputeFieldCallback(TestWithIntegrator, unittest.TestCase):
    """Tests hpmc.field.callback."""
    def build_object(self):
        return hpmc.field.callback(self.mc, lambda snapshot: 0)


class TestComputeFieldFrenkelLadd(TestWithIntegrator, unittest.TestCase):
    """Tests hpmc.field.frenkel_ladd_energy."""
    def build_object(self):
        snap = self.system.take_snapshot()
        # Note that while serialization of NumPy arrays does work, the test
        # case requires conversion to list so that the comparisons in the
        # parent class's test function work.
        rs = snap.particles.position.tolist()
        qs = snap.particles.orientation.tolist()
        return hpmc.field.frenkel_ladd_energy(
            self.mc, ln_gamma=0.0, q_factor=10.0, r0=rs, q0=qs, drift_period=1000)


class TestComputeFieldLattice(TestWithIntegrator, unittest.TestCase):
    """Tests hpmc.field.lattice_field."""
    def build_object(self):
        snap = self.system.take_snapshot()
        # Note that while serialization of NumPy arrays does work, the test
        # case requires conversion to list so that the comparisons in the
        # parent class's test function work.
        rs = snap.particles.position.tolist()
        qs = snap.particles.orientation.tolist()
        return hpmc.field.lattice_field(
            self.mc, position=rs, orientation=qs, k=10, q=10)

class TestComputeFieldWall(TestWithIntegrator, unittest.TestCase):
    """Tests hpmc.field.wall."""
    def build_object(self):
        import numpy as np
        ext_wall = hpmc.field.wall(self.mc)
        ext_wall.add_sphere_wall(radius=1.0, origin=[0, 0, 0], inside=True)
        ext_wall.set_volume(4./3.*np.pi)
        return ext_wall


class TestIntegrateConvexPolygon(TestMetadata, unittest.TestCase):
    """Tests hpmc.integrate.convex_polygon."""
    def build_object(self):
        return hpmc.integrate.convex_polygon(8093)


class TestIntegrateConvexPolyhedron(TestMetadata, unittest.TestCase):
    """Tests hpmc.integrate.convex_polyhedron."""
    def build_object(self):
        return hpmc.integrate.convex_polyhedron(8093)


class TestIntegrateConvexPolyhedronUnion(TestMetadata, unittest.TestCase):
    """Tests hpmc.integrate.convex_polyhedron_union."""
    def build_object(self):
        return hpmc.integrate.convex_polyhedron_union(8093)


class TestIntegrateConvexSpheropolygon(TestMetadata, unittest.TestCase):
    """Tests hpmc.integrate.convex_spheropolygon."""
    def build_object(self):
        return hpmc.integrate.convex_spheropolygon(8093)


class TestIntegrateConvexSpheropolyhedron(TestMetadata, unittest.TestCase):
    """Tests hpmc.integrate.convex_spheropolyhedron."""
    def build_object(self):
        return hpmc.integrate.convex_spheropolyhedron(8093)


class TestIntegrateConvexSpheropolyhedronUnion(TestMetadata, unittest.TestCase):
    """Tests hpmc.integrate.convex_spheropolyhedron_union."""
    def build_object(self):
        return hpmc.integrate.convex_spheropolyhedron_union(8093)


class TestIntegrateConvexEllipsoid(TestMetadata, unittest.TestCase):
    """Tests hpmc.integrate.ellipsoid."""
    def build_object(self):
        return hpmc.integrate.ellipsoid(8093)


class TestIntegrateConvexFacetedSphere(TestMetadata, unittest.TestCase):
    """Tests hpmc.integrate.faceted_sphere."""
    def build_object(self):
        return hpmc.integrate.faceted_sphere(8093)


class TestIntegratePolyhedron(TestMetadata, unittest.TestCase):
    """Tests hpmc.integrate.polyhedron."""
    def build_object(self):
        return hpmc.integrate.polyhedron(8093)


class TestIntegrateSimplePolygon(TestMetadata, unittest.TestCase):
    """Tests hpmc.integrate.simple_polygon."""
    def build_object(self):
        return hpmc.integrate.simple_polygon(8093)


class TestIntegrateSphere(TestMetadata, unittest.TestCase):
    """Tests hpmc.integrate.sphere."""
    def build_object(self):
        return hpmc.integrate.sphere(8093)


class TestIntegrateSphereUnion(TestMetadata, unittest.TestCase):
    """Tests hpmc.integrate.sphere_union."""
    def build_object(self):
        return hpmc.integrate.sphere_union(8093)


class TestIntegrateSphinx(TestMetadata, unittest.TestCase):
    """Tests hpmc.integrate.sphinx."""
    def build_object(self):
        return hpmc.integrate.sphinx(8093)


class TestUpdateBoxmc(TestWithIntegrator, unittest.TestCase):
    """Tests hpmc.update.boxmc."""
    def build_object(self):
        # Explicitly use variant._constant to avoid meaningless test failure.
        var = hoomd.variant._constant(10)
        bmc = hpmc.update.boxmc(self.mc, var, 13789)
        bmc.aspect(3, 0.3)
        bmc.length(2, 0.1)
        bmc.ln_volume(4, 0.6)
        return bmc


class TestUpdateUpdateClusters(TestWithIntegrator, unittest.TestCase):
    """Tests hpmc.update.clusters."""
    def build_object(self):
        clusters = hpmc.update.clusters(self.mc, 98986)
        clusters.set_params(swap_types=['A','A'], delta_mu = -0.001)
        return clusters


class TestUpdateUpdateMuvt(TestWithIntegrator, unittest.TestCase):
    """Tests hpmc.update.muvt."""
    def build_object(self):
        return hpmc.update.muvt(self.mc, 98986)


class TestUpdateUpdateRemoveDrift(TestWithIntegrator, unittest.TestCase):
    """Tests hpmc.update.remove_drift."""
    def build_object(self):
        snap = self.system.take_snapshot()
        # Note that while serialization of NumPy arrays does work, the test
        # case requires conversion to list so that the comparisons in the
        # parent class's test function work.
        rs = snap.particles.position.tolist()
        qs = snap.particles.orientation.tolist()
        lattice =  hpmc.field.lattice_field(
            self.mc, position=rs, orientation=qs, k=10, q=10)
        return hpmc.update.remove_drift(self.mc, lattice)


@unsupported
class TestComputeUpdateWall(TestWithIntegrator, unittest.TestCase):
    """Tests hpmc.update.wall."""
    def build_object(self):
        import numpy as np
        ext_wall = hpmc.field.wall(self.mc)
        ext_wall.add_sphere_wall(radius=1.0, origin=[0, 0, 0], inside=True)
        ext_wall.set_volume(4./3.*np.pi)
        def modify(timestep):
            ext_wall.set_sphere_wall(index=0, radius=1.0, origin=[0, 0, 0], inside=False);
        return hpmc.update.wall(self.mc, ext_wall, modify, move_ratio=0.1, seed=65738)


class TestUtilTune(TestWithIntegrator, unittest.TestCase):
    """Tests hpmc.util.tune"""
    def build_object(self):
        return hpmc.util.tune(
            self.mc, tunables=['a'], max_val=[1], target=0.3)


class TestUtilTuneNpt(TestWithIntegrator, unittest.TestCase):
    """Tests hpmc.util.tune_npt"""
    def build_object(self):
        # Explicitly use variant._constant to avoid meaningless test failure.
        var = hoomd.variant._constant(10)
        bmc = hpmc.update.boxmc(self.mc, var, 13789)
        bmc.length(0.1, weight=1)
        return hpmc.util.tune_npt(
            bmc, tunables=['dLx'], max_val=[1], target=0.3)


if __name__ == "__main__":
    unittest.main()
