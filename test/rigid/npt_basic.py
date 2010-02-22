from hoomd_script import *

init.read_xml('cubes.xml')

lj = pair.lj(r_cut=2**(1.0/6.0));
lj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
lj.set_params(mode='shift')
nlist.reset_exclusions(exclusions=['body'])

integrate.mode_standard(dt=0.005)
bdnvt = integrate.bdnvt_rigid(group=group.all(), T=1.2)

dcd = dump.dcd(filename='npt_basic.dcd', period=100, overwrite=True)
log = analyze.log(filename="npt_basic.log", period=100, overwrite=True, quantities=['potential_energy', 'kinetic_energy'])

run(5000)

bdnvt.disable()
npt = integrate.npt_rigid(group=group.all(), T=1.2, tau=1.0, P=1.0, tauP=2.5)

run(10000)

