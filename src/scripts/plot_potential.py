from hoomd_script import *

sysdef = init.create_empty(N=2, box = (20, 20, 20))
# parameters for the plot
rmin = 0.5
rmax = 5.0
dr = 0.001

# pair potential settings for the plot
lj = pair.lj(r_cut=2.5)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0)
#lj.set_params(mode='shift')

###########################################
## Do a run with the 2 paritlces, moving particle 1 a bit each time
## and plot the potential to a file after each step
# setup the particles for the run
p0 = sysdef.particles[0];
p1 = sysdef.particles[1];

p0.position = (0,0,0)
p0.velocity = (0,0,0)
p1.position = (rmin, 0, 0)
p1.velocity = (dr, 0, 0)

# setup the run to go between the desired rmin, rmax
nsteps = int((rmax - rmin) / dr);

integrate.mode_standard(dt=1.0)
integrate.nve(group=group.all(), zero_force=True)

# log the potential energy
logger = analyze.log(quantities=['potential_energy'], period=1, filename='full.log')

f = open('potential', 'w')

for i in xrange(0, nsteps):
    run(1)
    f.write("%f\t%f\n" % (p1.position[0], logger.query('potential_energy')))

