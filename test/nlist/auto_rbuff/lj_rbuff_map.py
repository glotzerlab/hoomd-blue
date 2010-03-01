from hoomd_script import *

f = open('lj_rbuff_tps', 'w');
f.write("r\tTPS\n");

init.create_random(N=10000, phi_p=0.3)
lj = pair.lj(r_cut=3.0)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0)

all = group.all()
integrate.mode_standard(dt=0.005)
integrate.nvt(group=all, T=0.1, tau=0.5)

run(10000)

for i in xrange(0, 11):
    r = 0.2 + float(i) * 1.0/10.0
    
    nlist.set_params(r_buff = r)
    run(1000)
    TPS = globals.system.getLastTPS();
    f.write("%f\t%f\n" % (r, TPS))
    f.flush();

for i in xrange(0, 11):
    r = 0.2 + float(i) * 1.0/10.0
    
    nlist.set_params(r_buff = r)
    run(1000)
    TPS = globals.system.getLastTPS();
    f.write("%f\t%f\n" % (r, TPS))
    f.flush();

for i in xrange(0, 11):
    r = 0.2 + float(i) * 1.0/10.0
    
    nlist.set_params(r_buff = r)
    run(1000)
    TPS = globals.system.getLastTPS();
    f.write("%f\t%f\n" % (r, TPS))
    f.flush();

f.close()
