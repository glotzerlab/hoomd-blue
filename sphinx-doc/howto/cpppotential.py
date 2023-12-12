import hoomd
import gsd.hoomd

frame = gsd.hoomd.Frame()

# Place particles in the box
frame.particles.N = 2
frame.particles.position = [[-0.51, 0, 0], [0.51, 0, 0]]
frame.particles.types = ['S']
frame.particles.typeid = [0] * 2
frame.configuration.box = [20, 20, 20, 0, 0, 0]

with gsd.hoomd.open(name='cpppotential.gsd', mode='x') as f:
    f.append(frame)

# The square well potential
r_interaction = 1.1
evaluate_square_well = f'''float rsq = dot(r_ij, r_ij);
                    if (rsq < {r_interaction * r_interaction}f)
                        return param_array[0];
                    else
                        return 0.0f;
            '''
square_well = hoomd.hpmc.pair.user.CPPPotential(r_cut=r_interaction,
                                                code=evaluate_square_well,
                                                param_array=[-1])

# hpmc.Sphere provides the hard sphere part of the potential
mc = hoomd.hpmc.integrate.Sphere()
mc.shape['S'] = dict(diameter=1.0)

# Compute the square potential when evaluating trial moves
mc.pair_potential = square_well

# Create the simulation.
sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
sim.create_state_from_gsd(filename='cpppotential.gsd')
sim.operations.integrator = mc

# The energy is -1
sim.run(0)
print(square_well.energy)

# Change potential parameters by setting param_array:
square_well.param_array[0] = -2
print(square_well.energy)
