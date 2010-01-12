from hoomd_script import *

init.read_xml('rods.xml')

lj = pair.lj(r_cut=2**(1.0/6.0));
lj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
lj.set_params(mode='shift')
nlist.reset_exclusions(exclusions=['body'])

integrate.mode_standard(dt=0.005)
integrate.nve_rigid(group=group.all())

dump.dcd(filename='nve.dcd', period=50, overwrite=True)
analyze.log(filename="nve.log", period=100, overwrite=True, quantities=['potential_energy', 'kinetic_energy'])

run(100000)
