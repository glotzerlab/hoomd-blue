from hoomd_script import *

init.read_xml('cubes.xml')

lj = pair.lj(r_cut=2**(1.0/6.0));
lj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
lj.set_params(mode='shift')
nlist.reset_exclusions(exclusions=['body'])

integrate.mode_standard(dt=0.005)
bdnvt = integrate.bdnvt_rigid(group=group.all(), T=1.2)

dcd = dump.dcd(filename='nve.dcd', period=100, overwrite=True)
log = analyze.log(filename="nve_basic.log", period=100, overwrite=True, quantities=['potential_energy', 'kinetic_energy'])

run(5000)

bdnvt.disable()
nve = integrate.nve_rigid(group=group.all())

run(5000)

xml = dump.xml()
xml.set_params(all=True)
xml.write(filename="continue.xml")

dcd = dump.dcd(filename='baseline.dcd', period=100, overwrite=True)

run(5000)
