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
import util

import math
import os
import pickle
import sys

##
# \package hoomd_script.tune
# \brief Commands for tuning the performance of HOOMD

###########################################################################
# Code for storing and retrieving the optimal block size for a specific command

## \internal
# \brief Default database of optimal block sizes
#
# Defaults are saved per compute capability and per command
_default_block_size_db = {};
_default_block_size_db['1.1'] = {'pair.gauss': 320, 'bond.fene': 256, 'pair.lj': 320, 'nlist': 320, 'bond.harmonic': 64};
_default_block_size_db['1.3'] = {'pair.gauss': 224, 'bond.fene': 288, 'pair.lj': 384, 'nlist': 128, 'bond.harmonic': 352}

## \internal
# \brief Optimal block size database user can load to override the defaults
_override_block_size_db = None;
_override_block_size_compute_cap = None;

## \internal
# \brief Saves the block size tuning override file
# 
# \param common_optimal_db Dictionary of the common optimal block sizes identified
# 
def _save_override_file(common_optimal_db):
	fname = os.getenv('HOME') + '/.hoomd_block_tuning';
	
	# see if the user really wants to overwrite the file
	if os.path.isfile(fname):
		print "\n***Warning!", fname, "exists. This file is being overwritten with new settings\n";	

	# save the file
	f = file(fname, 'w');
	print 'Writing optimal block sizes to', fname
	
	# write the version of the file
	pickle.dump(0, f);
	# write out the version this was tuned on
	pickle.dump(hoomd.get_hoomd_version(), f);
	# write out the compute capability of the GPU this was tuned on
	pickle.dump(globals.particle_data.getExecConf().getComputeCapability(), f);
	# write out the dictionary
	pickle.dump(common_optimal_db, f);
	
	f.close();
	
## \internal
# \brief Loads in the override block size tuning db
# 
# unpickles the file ~/.hoomd_block_tuning and fills out _override_block_size_db and _override_block_size_compute_cap
#
def _load_override_file():
	global _override_block_size_db, _override_block_size_compute_cap;
	
	fname = os.getenv('HOME') + '/.hoomd_block_tuning';
	
	# only load if the file exists
	if not os.path.isfile(fname):
		return

	# save the file
	f = file(fname, 'r');
	print 'Notice: Reading optimal block sizes from', fname
	
	# read the version of the file
	ver = pickle.load(f);
	
	# handle the different file versions
	if ver == 0:
		# read and verify the version this was tuned on
		hoomd_version = pickle.load(f);
		if hoomd_version != hoomd.get_hoomd_version():
			print >> sys.stderr, "\n***Warning! .hoomd_block_tuning was created with", hoomd_version, \
								", but this is", hoomd.get_hoomd_version(), ". Reverting to default performance tuning.\n";
			return;
		
		# read the compute capability of the GPU this was tuned on
		_override_block_size_compute_cap = pickle.load(f);
		# read the dictionary
		_override_block_size_db = pickle.load(f);
		
	else:
		print >> sys.stderr, "\n***Error! Unknown .hoomd_block_tuning format", ver, ".\n";
		raise RuntimeError("Error loading .hoomd_block_tuning");
	
	f.close();

# load the override file on startup
_load_override_file();

## \internal
# \brief Retrieves the optimal block size saved in the database
#
# \param name Name of the command to get the optimal block size for
#
# First, the user override db will be checked: if the value is present there
# it will be returned.
#
# If no override is specified, the optimal block size will be retrieved from
# the default database above
def _get_optimal_block_size(name):
	global _override_block_size_db, _override_block_size_compute_cap;
	
	compute_cap = globals.particle_data.getExecConf().getComputeCapability();
	
	# check for the override first
	if _override_block_size_db != None:
		# first verify the compute capability
		if compute_cap == _override_block_size_compute_cap:		
			if name in _override_block_size_db:
				return _override_block_size_db[name];
			else:
				print >> sys.stderr, "\n***Error! Block size override db does not contain a value for", name, ".\n";
				raise RuntimeError("Error retrieving optimal block size");
		else:
			print "\n***Warning! The compute capability of the current GPU is", compute_cap, "while the override was tuned on a", _override_block_size_compute_cap, "GPU"
			print "            Ignoring the saved override in ~/.hoomd_block_tuning and reverting to the default.\n"


	# check in the default db
	if compute_cap in _default_block_size_db:
		if name in _default_block_size_db[compute_cap]:
			return _default_block_size_db[compute_cap][name];
		else:
			print >> sys.stderr, "\n***Error! Default block size db does not contain a value for", name, ".\n";
			raise RuntimeError("Error retrieving optimal block size");
	else:
		print "\n***Warning! Optimal block size tuning values are not present for your hardware with compute capability", compute_cap;
		print "            To obtain better performance, execute the following hoomd script to determine the optimal"
		print "            settings and save them in your home directory. Future invocations of hoomd will use these"
		print "            saved values\n"
		print "            # block size tuning script"
		print "            from hoomd_script import *"
		print "            tune.find_optimal_block_sizes()\n"
		return 64


###########################################################################
# Code for tuning and saving the optimal block size

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

## \internal
# \brief Chooses a common optimal block sizes from a list of several dictionaries
#
# \param optimal_dbs List of dictionaries to choose common values from
#
def _choose_optimal_block_sizes(optimal_dbs):
	# create a new db with the common optimal settings
	common_optimal_db = {};
	
	# choose the most common optimal block size for each entry
	for entry in optimal_dbs[0]:
		
		count = {};
		
		for db in optimal_dbs:
			best_block_size = db[entry];
			count[best_block_size] = count.setdefault(best_block_size, 0) + 1;
		
		# find the most common
		common_optimal_list = [];
		max_count = max(count.values());
		for (block_size, c) in count.items():
			if c == max_count:
				common_optimal_list.append(block_size);
		
		# add it to the common optimal db
		common_optimal = common_optimal_list[0];
		common_optimal_db[entry] = common_optimal;
		
		if len(common_optimal_list) > 1:
			print "Notice: more than one common optimal block size found for", entry, ", using", common_optimal;
	
	return common_optimal_db;
	
## Determine optimal block size tuning parameters
#
# \param save Set to False to disable the saving of ~/.hoomd_block_tuning
#
def find_optimal_block_sizes(save = True):
	util._disable_status_lines = True;
	
	# list of force computes to tune
	fc_list = [	('pair.lj', '(r_cut=3.0)', 500),
		     	('pair.gauss', '(r_cut=3.0)', 500),
				('bond.harmonic', '()', 10000),
				('bond.fene', '()', 5000)
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
	
	# print out all the optimal block sizes
	print '*****************'
	print 'Optimal block sizes found: '
	for db in optimal_dbs:
		print db;
	
	# create a new db with the common optimal settings
	print "Chosing common optimal block sizes:"
	common_optimal_db = _choose_optimal_block_sizes(optimal_dbs);
	print common_optimal_db;
		
	print '*****************'
	print
	if save:
		_save_override_file(common_optimal_db);
	
	init.reset();
	util._disable_status_lines = False;
