from hoomd_script import *

init.read_bin('continue.bin.gz')

lj = pair.lj(r_cut=2**(1.0/6.0));
lj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
lj.set_params(mode='shift')
nlist.reset_exclusions(exclusions=['body'])

integrate.mode_standard(dt=0.005)
integrate.nve_rigid(group=group.all())

dcd = dump.dcd(filename='restart_bin.dcd', period=100, overwrite=True)
log = analyze.log(filename="restart_bin.log", period=1, overwrite=True, quantities=['potential_energy', 'kinetic_energy'])

run(50000)
