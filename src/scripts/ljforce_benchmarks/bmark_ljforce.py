#! /usr/bin/python

import os;

# for GTX 280
args_base = "--half_nlist=0 -q --block_size=96 --sort=0";

# for CPU
# args_base = "-q";

for N in xrange(2000,200000, 2000):
	os.system("./force_compute_bmark " + args_base + " -N " + str(N));
