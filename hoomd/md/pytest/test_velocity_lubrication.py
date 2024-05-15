# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest

import hoomd
import hoomd.conftest
from hoomd import md

def test_sheared_lubrication_forces(simulation_factory, two_particle_snapshot_factory):
    mu = 2.38
    rad_a, rad_b = 0.789, 3.2
    h = 0.108
    velocity_x = 15.03
    mass_1 = 5.33
    mass_2 = 5.1
    I_1 = mass_1 * rad_a * rad_a * 0.4
    I_2 = mass_2 * rad_b * rad_b * 0.4
    beta = rad_b/rad_a
    gam_inv = ((rad_a+rad_b)/2)/h

    lub_force = md.pair.aniso.VelocityLubricationCoupling(nlist = md.nlist.Cell(buffer=0.1),default_r_cut = 1.5,mode='none')
    lub_force.params[('A','A')] = dict(mu=mu,take_momentum=True,take_velocity=True)
    lub_force.params[('A','B')] = dict(mu=mu,take_momentum=True,take_velocity=True)
    lub_force.params[('B','B')] = dict(mu=mu,take_momentum=True,take_velocity=True)
    lub_force.r_cut[('A','A')] = 1.2 * (rad_a + rad_a)
    lub_force.r_cut[('A','B')] = 1.2 * (rad_a + rad_b)
    lub_force.r_cut[('B','B')] = 1.2 * (rad_b + rad_b)
    s = two_particle_snapshot_factory(particle_types=['A','B'],d=rad_a+rad_b+h, L = 50 )
    s.particles.mass[:] = [mass_1,mass_2]
    s.particles.moment_inertia[:] = [(I_1,I_1,I_1),(I_2,I_2,I_2)]
    s.particles.position[:] = [[0,0,rad_a+rad_b+h],[0,0,0]]
    s.particles.diameter[:] = [rad_a*2,rad_b*2]
    s.particles.typeid[:] = [0,1]
    s.particles.velocity[:] = np.array([[velocity_x,0,0],[0.,0,0]])
    s.particles.angmom[:] = np.array([[0,0,0,0],[0,0,0,0]])

    sim = simulation_factory(s)
    integrator = md.Integrator(0.005, forces=[lub_force],integrate_rotational_dof=True)
    langevin = hoomd.md.methods.Langevin(hoomd.filter.All(),kT = 0.0,default_gamma = 0.0, default_gamma_r=(0.0,0.0,0.0))
    integrator.forces.append(lub_force)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator
    sim.run(0)
    lub_F = lub_force.forces[0,:]
    lub_T = lub_force.torques[0,:]
    force_analytical = np.array([-6 * np.pi * mu * rad_a * velocity_x * (4*beta*(2+beta+2*beta*beta))/(15*(1+beta)**3)*np.log(gam_inv), 0, 0])
    torque_analytical = np.array([0., -8 * mu * np.pi * rad_a*rad_a * velocity_x * (4*beta+beta*beta)/(10*(1+beta)**2)*np.log(gam_inv), 0.])
    assert np.isclose(lub_F,force_analytical).all()
    assert np.isclose(lub_T,torque_analytical).all()

def test_sheared_rotation_lubrication_forces(simulation_factory, two_particle_snapshot_factory):
    mu = 2.38
    rad_a, rad_b = 0.789, 3.2
    h = 0.108
    ang_vel_y = 15.03
    mass_1 = 5.33
    mass_2 = 5.1
    I_1 = mass_1 * rad_a * rad_a * 0.4
    I_2 = mass_2 * rad_b * rad_b * 0.4
    beta = rad_b/rad_a
    gam_inv = ((rad_a+rad_b)/2)/h

    lub_force = md.pair.aniso.VelocityLubricationCoupling(nlist = md.nlist.Cell(buffer=0.1),default_r_cut = 1.5,mode='none')
    lub_force.params[('A','A')] = dict(mu=mu,take_momentum=True,take_velocity=True)
    lub_force.params[('A','B')] = dict(mu=mu,take_momentum=True,take_velocity=True)
    lub_force.params[('B','B')] = dict(mu=mu,take_momentum=True,take_velocity=True)
    lub_force.r_cut[('A','A')] = 1.2 * (rad_a + rad_a)
    lub_force.r_cut[('A','B')] = 1.2 * (rad_a + rad_b)
    lub_force.r_cut[('B','B')] = 1.2 * (rad_b + rad_b)
    s = two_particle_snapshot_factory(particle_types=['A','B'],d=rad_a+rad_b+h, L = 50 )
    s.particles.mass[:] = [mass_1,mass_2]
    s.particles.moment_inertia[:] = [(I_1,I_1,I_1),(I_2,I_2,I_2)]
    s.particles.position[:] = [[0,0,rad_a+rad_b+h],[0,0,0]]
    s.particles.diameter[:] = [rad_a*2,rad_b*2]
    s.particles.typeid[:] = [0,1]
    s.particles.velocity[:] = np.array([[0,0,0],[0.,0,0]])
    s.particles.angmom[:] = np.array([[0,0,2*ang_vel_y*I_1,0],[0,0,0,0]])

    sim = simulation_factory(s)
    integrator = md.Integrator(0.005, forces=[lub_force],integrate_rotational_dof=True)
    langevin = hoomd.md.methods.Langevin(hoomd.filter.All(),kT = 0.0,default_gamma = 0.0, default_gamma_r=(0.0,0.0,0.0))
    integrator.forces.append(lub_force)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator
    sim.run(0)
    lub_F = lub_force.forces[0,:]
    lub_T = lub_force.torques[0,:]
    force_analytical_x = 8 * np.pi * mu * rad_a * rad_a * ang_vel_y * (4 * beta+beta*beta)/(10*(1+beta)**2) * np.log(gam_inv)
    torque_analytical_y = -8 * np.pi* mu * rad_a**3 * ang_vel_y * (2 * beta)/(5+5*beta) * np.log(gam_inv)
    force_analytical = np.array([force_analytical_x, 0, 0])
    torque_analytical = np.array([0., torque_analytical_y, 0.])
    assert np.isclose(lub_F,force_analytical).all()
    assert np.isclose(lub_T,torque_analytical).all()
