from hoomd_script import *

init.read_xml('cubes.xml')

lj = pair.lj(r_cut=2**(1.0/6.0));
lj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
lj.set_params(mode='shift')
nlist.reset_exclusions(exclusions=['body'])
ljw = wall.lj(r_cut=3.0)
ljw.set_coeff('A', epsilon=1.0, sigma=1.0, alpha=0.0)

integrate.mode_standard(dt=0.005)
bdnvt = integrate.bdnvt_rigid(group=group.all(), T=1.2)

dcd = dump.dcd(filename='bdnvt_basic.dcd', period=100, overwrite=True)
log = analyze.log(filename="bdnvt_basic.log", period=100, overwrite=True, quantities=['potential_energy', 'kinetic_energy'])

run(1e6)
