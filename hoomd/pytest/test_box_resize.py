from math import isclose
import numpy as np
import pytest
import hoomd


class SimulationSetup:
    def __init__(self, box1_dim, box2_dim, n_particles=10,
                 t_start=5, t_ramp=10):
        self.n_particles = int(n_particles)
        self.t_start = int(t_start)
        self.t_ramp = int(t_ramp)

        self.box1 = hoomd.Box.from_box(box1_dim)
        self.box2 = hoomd.Box.from_box(box2_dim)

        points = np.random.uniform(-0.5, 0.5, size=(self.n_particles, 3))
        self.points1 = points @ self.box1.matrix.T
        self.points2 = points @ self.box2.matrix.T

    def get_snapshot(self):
        snap = hoomd.Snapshot()
        snap.configuration.box = self.box1
        snap.particles.N = self.n_particles
        snap.particles.typeid[:] = [0] * self.n_particles
        snap.particles.types = ['A']
        snap.particles.position[:] = self.points1
        return snap

    def variant_ramp(self, scale_particles=True):
        sim = hoomd.Simulation(device)
        sim.create_state_from_snapshot(self.get_snapshot())

        variant = hoomd.variant.Ramp(A=0., B=1., t_start=self.t_start, t_ramp=self.t_ramp)
        trigger = hoomd.trigger.On(self.t_start)
        box_resize = hoomd.update.BoxResize(
            box1=self.box1, box2=self.box2,
            variant=variant, trigger=trigger, scale_particles=scale_particles)
        sim.operations.updaters.append(box_resize)
        sim.run(self.t_start + self.t_ramp + 1)
        return box_resize, sim.state.snapshot

    def variant_linear(self, scale_particles=True):
        sim = hoomd.Simulation(device)
        sim.create_state_from_snapshot(self.get_snapshot())

        trigger = hoomd.trigger.On(self.t_start)
        box_resize = hoomd.update.BoxResize.linear_volume(
            box1=self.box1, box2=self.box2,
            t_start=self.t_start, t_size=self.t_ramp,
            trigger=trigger, scale_particles=scale_particles)
        sim.operations.updaters.append(box_resize)
        sim.run(self.t_start + self.t_ramp + 1)
        return box_resize, sim.state.snapshot


@pytest.mark.parametrize("box1_dim, box2_dim", 
  [([1., 2., 3., 1., 2., 3.], [0.5, 2., 3., 1., 2., 3.]),
   ([1., 2., 3., 1., 2., 3.], [1., 2., 3., 0., 2., 3.]),
   ([1., 2., 3., 1., 2., 3.], [10, 2., 3., 0, 2., 3.]),
   ([1., 2., 3., 1., 2., 3.], [10., 1., 6., 0., 5., 7.]),])
def test_variant_ramp(box1_dim, box2_dim):
    ss = SimulationSetup(box1_dim=box1_dim, box2_dim=box2_dim)
    box_resize, snap = ss.variant_ramp()

    assert np.all(box_resize.get_box(0).matrix == ss.box1.matrix)
    assert np.all(box_resize.get_box(ss.t_start + ss.t_ramp).matrix == ss.box2.matrix)
    # assert np.all(np.isclose(snap.particles.position[:], ss.points2))


@pytest.mark.parametrize("box1_dim, box2_dim", 
  [([1., 2., 3., 1., 2., 3.], [0.5, 2., 3., 1., 2., 3.]),
   ([1., 2., 3., 1., 2., 3.], [1., 2., 3., 0., 2., 3.]),
   ([1., 2., 3., 1., 2., 3.], [10, 2., 3., 0, 2., 3.]),
   ([1., 2., 3., 1., 2., 3.], [10., 1., 6., 0., 5., 7.]),])
def test_variant_linear(box1_dim, box2_dim):
    ss = SimulationSetup(box1_dim=box1_dim, box2_dim=box2_dim)
    box_resize, snap = ss.variant_linear()

    assert np.all(box_resize.get_box(0).matrix == ss.box1.matrix)
    assert np.all(box_resize.get_box(ss.t_start + ss.t_ramp).matrix == ss.box2.matrix)
    # assert np.all(np.isclose(snap.particles.position[:], ss.points2))
