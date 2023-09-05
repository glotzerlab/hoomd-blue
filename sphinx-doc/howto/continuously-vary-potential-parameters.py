import hoomd

# Preparation: Create a MD simulation.
simulation = hoomd.util.make_example_simulation()

lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4))
lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
lj.r_cut[('A', 'A')] = 2.5

langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.5)
simulation.operations.integrator = hoomd.md.Integrator(dt=0.001,
                                                       methods=[langevin],
                                                       forces=[lj])


# Step 1: Subclass hoomd.custom.Action.
class LJParameterModifer(hoomd.custom.Action):

    def __init__(self, lj):
        super().__init__()
        self.lj = lj

    def act(self, timestep):
        epsilon = 1.0 + 4.0 * timestep / 1e6
        self.lj.params[('A', 'A')] = dict(epsilon=epsilon, sigma=1)


# Step 2: Create a hoomd.update.CustomUpdater
lj_parameter_modifier = LJParameterModifer(lj)
lj_parameter_updater = hoomd.update.CustomUpdater(
    trigger=hoomd.trigger.Periodic(1), action=lj_parameter_modifier)

# Step 3: Add the updater to the operations
simulation.operations.updaters.append(lj_parameter_updater)

simulation.run(100)
