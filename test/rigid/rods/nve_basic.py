from hoomd_script import *

init.read_xml('rods.xml')

lj = pair.lj(r_cut=2**(1.0/6.0));
lj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
lj.set_params(mode='shift')
nlist.reset_exclusions(exclusions=['body'])

integrate.mode_standard(dt=0.005)
bdnvt = integrate.bdnvt_rigid(group=group.all(), T=1.2)

dump.dcd(filename='nve_basic.dcd', period=50, overwrite=True)
analyze.log(filename="nve_basic.log", period=100, overwrite=True, quantities=['potential_energy', 'kinetic_energy'])

run(5000)

bdnvt.disable()
nve = integrate.nve_rigid(group=group.all())

run(100000)
