import unittest
import tempfile
import json
import warnings

import hoomd
from hoomd import md

from hoomd_test_metadata import TestMetadata, TestWithFile, unsupported

class TestMetadataMD(TestMetadata):
    """Extend the base class to provide a system with all the necessary
    components for testing."""
    def get_box(self):
        """Avoid recreating the box, which convolutes the metadata output."""
        try:
            return self.box
        except AttributeError:
            self.L = 10
            self.box = hoomd.data.boxdim(L=self.L)
            return self.box

    def make_system(self):
        box = self.get_box()
        positions = [box.wrap([i, j, k])[0] for i in range(self.L)
                                            for j in range(self.L)
                                            for k in range(self.L)]
        snap = hoomd.data.make_snapshot(
            N=len(positions),
            angle_types=['angle'],
            bond_types=['bond'],
            dihedral_types=['dihedral'],
            improper_types=['improper'],
            pair_types=['pair'],
            box=box)
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:] = positions
            snap.particles.charge[:] = 1
            snap.angles.resize(1)
            snap.angles.group[:] = [0, 1, 2]
            snap.bonds.resize(1)
            snap.bonds.group[:] = [3, 4]
            snap.dihedrals.resize(1)
            snap.dihedrals.group[:] = [5, 6, 7, 8]
            snap.impropers.resize(1)
            snap.impropers.group[:] = [9, 10, 11, 12]
            snap.pairs.resize(1)
            snap.pairs.group[:] = [13, 14]
            snap.constraints.resize(1)
            snap.constraints.group[:] = [15, 16]
        return hoomd.init.read_snapshot(snap)


class TestAngleHarmonic(TestMetadataMD, unittest.TestCase):
    """Tests md.angle.harmonic."""
    def build_object(self):
        angle = md.angle.harmonic()
        angle.angle_coeff.set('angle', k=3.0, t0=0.5)
        return angle


class TestAngleCosinesq(TestMetadataMD, unittest.TestCase):
    """Tests md.angle.cosinesq."""
    def build_object(self):
        angle = md.angle.cosinesq()
        angle.angle_coeff.set('angle', k=3.0, t0=0.5)
        return angle


@unsupported
class TestAngleTable(TestMetadataMD, unittest.TestCase):
    """Tests md.angle.table."""
    def build_object(self):
        def harmonic(theta, kappa, theta_0):
            V = 0.5 * kappa * (theta-theta_0)**2;
            T = -kappa*(theta-theta_0);
            return (V, T)

        angle = md.angle.table(width=1000)
        angle.angle_coeff.set('angle', func=harmonic, coeff=dict(kappa=330, theta_0=0))
        return angle


class TestBondFene(TestMetadataMD, unittest.TestCase):
    """Tests md.bond.fene."""
    def build_object(self):
        bond = md.bond.fene()
        bond.bond_coeff.set('bond', k=30.0, r0=1.5, sigma=1.0, epsilon= 2.0)
        return bond


class TestBondHarmonic(TestMetadataMD, unittest.TestCase):
    """Tests md.bond.harmonic."""
    def build_object(self):
        bond = md.bond.harmonic()
        bond.bond_coeff.set('bond', k=30.0, r0=1.5, sigma=1.0, epsilon= 2.0)
        return bond


@unsupported
class TestBondTable(TestMetadataMD, unittest.TestCase):
    """Tests md.bond.table."""
    def build_object(self):
        def harmonic(r, rmin, rmax, kappa, r0):
           V = 0.5 * kappa * (r-r0)**2;
           F = -kappa*(r-r0);
           return (V, F)

        bond = md.bond.table(width=1000)
        bond.bond_coeff.set('bond', func=harmonic, rmin=0.2, rmax=5.0, coeff=dict(kappa=330, r0=0.84))
        return bond


@unsupported
class TestChargePPPM(TestMetadataMD, unittest.TestCase):
    """Tests md.charge.pppm."""
    def build_object(self):
        charged = hoomd.group.charged()
        nlist = md.nlist.tree()
        pppm = md.charge.pppm(charged, nlist)
        pppm.set_params(Nx=2, Ny=2, Nz=2, order=2, rcut=2.0)
        return pppm


class TestConstraintDistance(TestMetadataMD, unittest.TestCase):
    """Tests md.constrain.distance."""
    def build_object(self):
        constraint = md.constrain.distance()
        constraint.set_params(rel_tol=0.0001)
        return constraint


@unsupported
class TestConstraintRigid(TestMetadataMD, unittest.TestCase):
    """Tests md.constrain.rigid."""
    def build_object(self):
        self.system.particles.types.add('B')
        constraint = md.constrain.rigid()
        constraint.set_param('A',
                        types=['B']*8,
                        positions=[(-4,0,0),(-3,0,0),(-2,0,0),(-1,0,0),
                                   (1,0,0),(2,0,0),(3,0,0),(4,0,0)]);
        return constraint


class TestConstraintSphere(TestMetadataMD, unittest.TestCase):
    """Tests md.constrain.sphere."""
    def build_object(self):
        group = hoomd.group.tags(0)
        constraint = md.constrain.sphere(group, [-1, 0, 0], 1)
        return constraint


class TestConstraintOned(TestMetadataMD, unittest.TestCase):
    """Tests md.constrain.oneD."""
    def build_object(self):
        group = hoomd.group.all()
        constraint = md.constrain.oneD(group, constraint_vector=[1,0,0])
        return constraint


class TestDihedralOpls(TestMetadataMD, unittest.TestCase):
    """Tests md.dihedral.opls."""
    def build_object(self):
        dihedral = md.dihedral.opls()
        dihedral.dihedral_coeff.set('dihedral', k1=30.0, k2=15.5, k3=2.2, k4=23.8)
        return dihedral


class TestDihedralHarmonic(TestMetadataMD, unittest.TestCase):
    """Tests md.dihedral.harmonic."""
    def build_object(self):
        dihedral = md.dihedral.harmonic()
        dihedral.dihedral_coeff.set('dihedral', k=30.0, d=-1, n=3)
        return dihedral


@unsupported
class TestDihedralTable(TestMetadataMD, unittest.TestCase):
    """Tests md.dihedral.table."""
    def build_object(self):
        def harmonic(theta, kappa, theta0):
           V = 0.5 * kappa * (theta-theta0)**2;
           F = -kappa*(theta-theta0);
           return (V, F)

        dihedral = md.dihedral.table(width=1000)
        dihedral.dihedral_coeff.set('dihedral', func=harmonic, coeff=dict(kappa=330, theta_0=0.0))
        return dihedral


class TestExternalPeriodic(TestMetadataMD, unittest.TestCase):
    """Tests md.external.periodic."""
    def build_object(self):
        external = md.external.periodic()
        external.force_coeff.set('A', A=1.0, i=0, w=0.02, p=3)
        return external


class TestExternalEfield(TestMetadataMD, unittest.TestCase):
    """Tests md.external.e_field."""
    def build_object(self):
        return md.external.e_field([1,0,0])


if __name__ == "__main__":
    unittest.main()









