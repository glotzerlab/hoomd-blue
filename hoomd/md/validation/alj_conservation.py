import unittest

import numpy as np
import numpy.testing as npt
import tempfile

import hoomd
from hoomd import md
import gsd.hoomd


class TestALJ(unittest.TestCase):
    def test_conservation(self):
        # Initialize context
        hoomd.context.initialize()

        polygon_type = "Polygon"
        # For test, use a unit area hexagon.
        particle_vertices = np.array([[ 6.20403239e-01,  0.00000000e+00],
                                      [ 3.10201620e-01,  5.37284966e-01],
                                      [-3.10201620e-01,  5.37284966e-01],
                                      [-6.20403239e-01,  7.59774841e-17],
                                      [-3.10201620e-01, -5.37284966e-01],
                                      [ 3.10201620e-01, -5.37284966e-01]])
        poly_area = 1
        num_sides = len(particle_vertices)

        circumsphere_radius = particle_vertices[0, 0]
        circumsphere_diameter = 2*circumsphere_radius
        side_length = np.linalg.norm(particle_vertices[1] -
                                     particle_vertices[2])
        insphere_diameter = 2 * (
            side_length / (2*np.tan(np.pi/num_sides)))

        # Just initialize in a simple cubic lattice.
        system = hoomd.init.create_lattice(hoomd.lattice.sq(
            circumsphere_diameter*2, type_name=polygon_type), n=10)

        # Initialize moments of inertia since original simulation was
        # HPMC.
        mass = poly_area
        # https://math.stackexchange.com/questions/2004798/moment-of-inertia-for-a-n-sided-regular-polygon # noqa
        moment_inertia = (mass*circumsphere_diameter**2/6)*(
            1+2*np.cos(np.pi/num_sides)**2)
        for p in system.particles:
            p.mass = mass
            p.moment_inertia = [0, 0, moment_inertia]
            # Not sure if this should be insphere or circumsphere;
            # probably doesn't matter based on current usage, but may
            # matter in the future for the potential if it's modified
            # to actually use diameter.
            p.diameter = circumsphere_diameter

        nl = md.nlist.cell()

        packing_fraction = 0.4
        L_current = system.box.Lx
        final_area = poly_area*len(
            system.particles)/packing_fraction
        L_final = np.sqrt(final_area)

        n_comp_start = 1e4
        n_comp_end = 1e5
        n_comp = n_comp_end - n_comp_start
        n_total = 1e6

        hoomd.update.box_resize(
                L=hoomd.variant.linear_interp(
                    [(n_comp_start, L_current),
                     (n_comp_end, L_final)]),
                period=n_comp/10000,
                scale_particles=True)

        # Define ALJ potential
        alpha = 0  # Make it WCA-only (no attraction)
        eps_att = 1.0
        r_cut_scale = 1.3
        # insphere_radius = circumsphere_radius*cos(180/n)
        # kernel_scale = circumsphere_radius/insphere_radius
        # I thought this would be enough, but looks like you need quite
        # a bit larger of a cutoff, so I just multiply the kernel_scale
        # by 2.
        kernel_scale = 2 * (1/np.cos(np.pi/num_sides))
        r_cut_set = insphere_diameter*kernel_scale*r_cut_scale

        alj = md.pair.alj(r_cut=r_cut_set, nlist=nl)
        alj.shape[polygon_type] = list(particle_vertices)
        alj.pair_coeff.set(
            polygon_type,
            polygon_type,
            epsilon=eps_att,
            sigma_i=insphere_diameter,
            sigma_j=insphere_diameter,
            alpha=alpha)

        # Set up the system
        dt = 1e-4
        kT = 0.3
        md.integrate.mode_standard(dt=dt, aniso=True)
        group = hoomd.group.all()
        integrator = md.integrate.nve(group=group)
        integrator.randomize_velocities(kT, 43)

        dump_steps = 1000

        tmp_gsd = tempfile.mkstemp(suffix='.test.gsd')
        gsd_file = tmp_gsd[1]
        tmp_log = tempfile.mkstemp(suffix='.test.log')
        log_file = tmp_log[1]

        hoomd.analyze.log(
            filename=log_file,
            quantities=['potential_energy', 'kinetic_energy'],
            period=dump_steps,
            phase=0,
            overwrite=True)
        gsd = hoomd.dump.gsd(
            filename=gsd_file,
            period=dump_steps,
            phase=0,
            group=group,
            overwrite=True)
        gsd.dump_shape(alj)

        hoomd.run(n_comp + n_comp_start)

        # Add quick test of shape.
        npt.assert_allclose(alj.get_type_shapes()[0]['vertices'], particle_vertices, 1e-5)

        # Reset velocities after the compression.
        integrator.randomize_velocities(kT, 44)
        hoomd.run_upto(n_total)

        log_data = np.genfromtxt(log_file, names=True)
        equilibrated_data = log_data[log_data['timestep'] >= n_comp_end]
        total_energy = equilibrated_data['potential_energy'] + equilibrated_data['kinetic_energy']

        # Ensure energy conservation up to the 4 digit per-particle.
        self.assertLess(np.std(total_energy)/len(system.particles), 1e-4)

        # Test momentum conservation.
        velocities = []
        with gsd.hoomd.open(gsd_file, 'rb') as traj:
            for frame in traj:
                velocities.append(frame.particles.velocity)
        velocities = np.asarray(velocities)
        self.assertLess(np.std(np.linalg.norm(np.sum(velocities, axis=1), axis=-1)), 1e-6)


if __name__ == "__main__":
    unittest.main(argv = ['test.py', '-v'])
