# Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
# Source Software License
# Copyright (c) 2008 Ames Laboratory Iowa State University
# All rights reserved.

# Redistribution and use of HOOMD, in source and binary forms, with or
# without modification, are permitted, provided that the following
# conditions are met:

# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names HOOMD's
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
# CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

# $Id$
# $URL$

import globals
import bond
import pair
import init
import hoomd_script
import hoomd
import math
import util

##
# \package hoomd_script.tune
# \brief Commands for tuning the performance of HOOMD

## \internal
# \brief Finds the optimal block size for a given force compute
# 
# \param fc Force compute to find the optimal block size of
# \param n Number of benchmark iterations to perform
# \return Fastest block size
#
# \note This function prints out status as it runs
#
def _find_optimal_block_size_fc(fc, n):
	timings = [];
	hoomd.set_gpu_error_checking(True)
	
	# run the benchmark
	try:
		for block_size in xrange(64,512,32):
			fc.cpp_force.setBlockSize(block_size);
			t = fc.benchmark(n);
			print block_size, t
			timings.append( (t, block_size) );
	except RuntimeError:
		print "Note: Too many resources requested for launch is a normal message when finding optimal block sizes"
		print ""
	
	fastest = min(timings);
	print 'fastest:', fastest[1]
	print '---------------'
	
	return fastest[1];
	
## \internal
# \brief Finds the optimal block size for a neighbor list compute
# 
# \param nl Force compute to find the optimal block size of
# \param n Number of benchmark iterations to perform
# \return Fastest block size
#
# \note This function prints out status as it runs
#
def _find_optimal_block_size_nl(nl, n):
	timings = [];
	hoomd.set_gpu_error_checking(True)
	
	# run the benchmark
	try:
		for block_size in xrange(64,544,32):
			nl.cpp_nlist.setBlockSize(block_size);
			t = nl.benchmark(n);
			print block_size, t
			timings.append( (t, block_size) );
	except RuntimeError:
		print "Note: Too many resources requested for launch is a normal message when finding optimal block sizes"
	
	
	fastest = min(timings);
	print 'fastest:', fastest[1]
	print '---------------'
	nl.cpp_nlist.setBlockSize(fastest[1]);
	
	return fastest[1];
		
## Determin optimal block size tuning parameters
#
# 
def find_optimal_block_sizes():
	util._disable_status_lines = True;
	
	# list of force computes to tune
	fc_list = [	('pair.lj', '(r_cut=3.0)', 500),
		     	('pair.gauss', '(r_cut=3.0)', 500),
				('bond.harmonic', '()', 10000),
				('bond.fene', '()', 10000)
				];
	
	# setup the particle system to benchmark
	polymer = dict(bond_len=1.2, type=['A']*50, bond="linear", count=2000);
	N = len(polymer['type']) * polymer['count'];
	phi_p = 0.2;
	L = math.pow(math.pi * N / (6.0 * phi_p), 1.0/3.0);

	init.create_random_polymers(box=hoomd.BoxDim(L), polymers=[polymer], separation=dict(A=0.35, B=0.35), seed=12)
	hoomd_script.run(1);
	
	# list of optimal databases
	optimal_dbs = [];
	num_repeats = 4;
	for i in xrange(0,num_repeats):
		
		# initialize an empty database of optimal sizes
		optimal_db = {};
		
		# for each force compute
		for (fc_name,fc_args,n) in fc_list:
			print 'Benchmarking ', fc_name
			# create it and benchmark it
			fc = eval(fc_name + fc_args)
			optimal = _find_optimal_block_size_fc(fc, n)
			optimal_db[fc_name] = optimal;
			
			# clean up
			fc.disable()
			del fc
		
		# now, benchmark the neighbor list
		print 'Benchmarking nlist'
		optimal = _find_optimal_block_size_nl(globals.neighbor_list, 100)
		optimal_db['nlist'] = optimal;
		
		# add it to the list
		optimal_dbs.append(optimal_db);
	
	init.reset();
	util._disable_status_lines = False;
	
	print '*****************'
	print 'Optimal block sizes found: '
	for db in optimal_dbs:
		print db;
	print '*****************'
	
	