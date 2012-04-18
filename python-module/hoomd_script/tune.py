# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
# Iowa State University and The Regents of the University of Michigan All rights
# reserved.

# HOOMD-blue may contain modifications ("Contributions") provided, and to which
# copyright is held, by various Contributors who have granted The Regents of the
# University of Michigan the right to modify and/or distribute such Contributions.

# You may redistribute, use, and create derivate works of HOOMD-blue, in source
# and binary forms, provided you abide by the following conditions:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions, and the following disclaimer both in the code and
# prominently in any materials provided with the distribution.

# * Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions, and the following disclaimer in the documentation and/or
# other materials provided with the distribution.

# * All publications and presentations based on HOOMD-blue, including any reports
# or published results obtained, in whole or in part, with HOOMD-blue, will
# acknowledge its use according to the terms posted at the time of submission on:
# http://codeblue.umich.edu/hoomd-blue/citations.html

# * Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
# http://codeblue.umich.edu/hoomd-blue/

# * Apart from the above required attributions, neither the name of the copyright
# holder nor the names of HOOMD-blue's contributors may be used to endorse or
# promote products derived from this software without specific prior written
# permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
# WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -- end license --
# Maintainer: joaander

import globals
import pair
import bond
import angle
import dihedral
import improper
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
_default_block_size_db['1.1'] = {'pair.ewald': 224, 'improper.harmonic': 64, 'pair.dpd_conservative': 320, 'dihedral.harmonic': 128, 'pair.dpd': 192, 'pair.dpdlj': 192, 'angle.cgcmm': 128, 'nlist.filter': 192, 'pair.lj': 320, 'pair.table': 320, 'bond.bondtable': 320, 'pair.cgcmm': 320, 'pair.slj': 256, 'pair.morse': 320, 'nlist': 288, 'bond.harmonic': 64, 'pair.yukawa': 320, 'bond.fene': 128, 'angle.harmonic': 192, 'pair.gauss': 320, 'external.periodic':256}

# no longer independently tuning 1.0 devices, they are very old
_default_block_size_db['1.0'] = _default_block_size_db['1.1'];

_default_block_size_db['1.3'] = {'pair.ewald': 160, 'improper.harmonic': 320, 'pair.dpd_conservative': 352, 'dihedral.harmonic': 256, 'pair.dpd': 320, 'angle.cgcmm': 64, 'nlist.filter': 160, 'pair.lj': 352, 'pair.table': 96, 'bond.bondtable': 96,'pair.cgcmm': 352, 'pair.dpdlj': 320, 'pair.slj': 352, 'pair.morse': 352, 'nlist': 416, 'bond.harmonic': 416, 'pair.yukawa': 352, 'bond.fene': 160, 'angle.harmonic': 192, 'pair.gauss': 352}

# no 1.2 devices to tune on. Assume the same as 1.3
_default_block_size_db['1.2'] = _default_block_size_db['1.3'];

_default_block_size_db['2.0'] = {'pair.ewald': 320, 'improper.harmonic': 96, 'pair.dpd_conservative': 224, 'dihedral.harmonic': 64, 'pair.dpd': 160, 'angle.cgcmm': 96, 'nlist.filter': 320, 'pair.lj': 320, 'pair.table': 128, 'bond.bondtable': 128, 'pair.cgcmm': 128, 'pair.dpdlj': 160, 'pair.slj': 160, 'pair.morse': 256, 'nlist': 768, 'bond.harmonic': 352, 'pair.yukawa': 320, 'bond.fene': 96, 'angle.harmonic': 128, 'pair.gauss': 320, 'external.periodic': 512}

_default_block_size_db['2.1'] = {'pair.ewald': 224, 'improper.harmonic': 96, 'pair.dpd_conservative': 224, 'dihedral.harmonic': 64, 'pair.dpd': 128, 'angle.cgcmm': 96, 'nlist.filter': 256, 'pair.lj': 160, 'pair.table': 160, 'bond.bondtable': 160, 'pair.cgcmm': 128, 'pair.dpdlj': 128, 'pair.slj': 128, 'pair.morse': 256, 'nlist': 576, 'bond.harmonic': 160, 'pair.yukawa': 192, 'bond.fene': 96, 'angle.harmonic': 96, 'pair.gauss': 160, 'external.periodic': 512}

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
    fname = os.path.expanduser("~") + '/.hoomd_block_tuning';
    
    # see if the user really wants to overwrite the file
    if os.path.isfile(fname):
        globals.msg.warning(fname + " exists. This file is being overwritten with new settings\n");

    # save the file
    f = file(fname, 'w');
    globals.msg.notice(2, 'Writing optimal block sizes to ' + str(fname) + '\n');
    
    # write the version of the file
    pickle.dump(0, f);
    # write out the version this was tuned on
    pickle.dump(hoomd.get_hoomd_version(), f);
    # write out the compute capability of the GPU this was tuned on
    pickle.dump(globals.system_definition.getParticleData().getExecConf().getComputeCapability(), f);
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
    
    fname = os.path.expanduser("~") + '/.hoomd_block_tuning';
    
    # only load if the file exists
    if not os.path.isfile(fname):
        return

    # save the file
    f = file(fname, 'r');
    globals.msg.notice(2, 'Reading optimal block sizes from ' + str(fname) + '\n');
    
    # read the version of the file
    ver = pickle.load(f);
    
    # handle the different file versions
    if ver == 0:
        # read and verify the version this was tuned on
        hoomd_version = pickle.load(f);
        if hoomd_version != hoomd.get_hoomd_version():
            globals.system.warning("~/.hoomd_block_tuning was created with" + str(hoomd_version) + \
                                ", but this is " + str(hoomd.get_hoomd_version()) + ". Reverting to default performance tuning.\n");
            return;
        
        # read the compute capability of the GPU this was tuned on
        _override_block_size_compute_cap = pickle.load(f);
        # read the dictionary
        _override_block_size_db = pickle.load(f);
        
    else:
        globals.msg.error("Unknown ~/.hoomd_block_tuning format " + str(ver) + ".\n");
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
    
    compute_cap = globals.system_definition.getParticleData().getExecConf().getComputeCapability();
    
    # check for the override first
    if _override_block_size_db is not None:
        # first verify the compute capability
        if compute_cap == _override_block_size_compute_cap:
            if name in _override_block_size_db:
                return _override_block_size_db[name];
            else:
                globals.msg.error("Block size override db does not contain a value for " + str(name) + ".\n");
                raise RuntimeError("Error retrieving optimal block size");
        else:
            globals.msg.warning("The compute capability of the current GPU is " + str(compute_cap) + " while the override was tuned on a " + str(_override_block_size_compute_cap) + " GPU\n");
            globals.msg.warning("Ignoring the saved override in ~/.hoomd_block_tuning and reverting to the default.\n");


    # check in the default db
    if compute_cap in _default_block_size_db:
        if name in _default_block_size_db[compute_cap]:
            return _default_block_size_db[compute_cap][name];
        else:
            globals.msg.error("Default block size db does not contain a value for " + str(name) + ".\n");
            raise RuntimeError("Error retrieving optimal block size");
    else:
        globals.msg.warning("Optimal block size tuning values are not present for your hardware with compute capability " + str(compute_cap) + "\n");
        globals.msg.warning("To obtain better performance, execute the following hoomd script to determine the optimal\n");
        globals.msg.warning("settings and save them in your home directory. Future invocations of hoomd will use these\n");
        globals.msg.warning("saved values\n");
        globals.msg.warning("# block size tuning script\n");
        globals.msg.warning("from hoomd_script import *\n");
        globals.msg.warning("tune.find_optimal_block_sizes()\n");
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
    
    # run the benchmark
    try:
        for block_size in xrange(64,1024+32,32):
            fc.cpp_force.setBlockSize(block_size);
            t = fc.benchmark(n);
            globals.msg.notice(2, str(block_size) + str(t) + '\n');
            timings.append( (t, block_size) );
    except RuntimeError:
        globals.msg.notice(2, "Note: Too many resources requested for launch is a normal message when finding optimal block sizes\n");

    fastest = min(timings);
    globals.msg.notice(2, 'fastest: ' + str(fastest[1]) + '\n');
    globals.msg.notice(2, '---------------\n');
    
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
    
    # run the benchmark
    try:
        for block_size in xrange(64,1024+32,32):
            nl.cpp_nlist.setBlockSize(block_size);
            t = nl.benchmark(n);
            globals.msg.notice(2, str(block_size) + ' ' + str(t) + '\n');
            timings.append( (t, block_size) );
    except RuntimeError:
        globals.msg.notice(2, "Note: Too many resources requested for launch is a normal message when finding optimal block sizes\n");
    
    
    fastest = min(timings);
    globals.msg.notice(2, 'fastest: ' + str(fastest[1]) + '\n');
    globals.msg.notice(2, '---------------\n');
    nl.cpp_nlist.setBlockSize(fastest[1]);
    
    return fastest[1];

## \internal
# \brief Finds the optimal block size for the neighbor list filter step
# 
# \param nl Neighbor list compute to find the optimal block size of
# \param n Number of benchmark iterations to perform
# \return Fastest block size
#
# \note This function prints out status as it runs
#
def _find_optimal_block_size_nl_filter(nl, n):
    timings = [];
    
    # run the benchmark
    try:
        for block_size in xrange(64,1024+32,32):
            nl.cpp_nlist.setBlockSizeFilter(block_size);
            t = nl.cpp_nlist.benchmarkFilter(n);
            globals.msg.notice(2, str(block_size) + ' ' + str(t) + '\n');
            timings.append( (t, block_size) );
    except RuntimeError:
        globals.msg.notice(2, "Note: Too many resources requested for launch is a normal message when finding optimal block sizes\n");
    
    
    fastest = min(timings);
    globals.msg.notice(2, 'fastest: ' + str(fastest[1]) + '\n');
    globals.msg.notice(2, '---------------\n');
    nl.cpp_nlist.setBlockSizeFilter(fastest[1]);
    
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
            globals.msg.notice(2, "More than one common optimal block size found for " + str(entry) + ", using" + str(common_optimal) + '\n');
    
    return common_optimal_db;

## Determine optimal block size tuning parameters
#
# \param save Set to False to disable the saving of ~/.hoomd_block_tuning
# \param only List of commands to limit the benchmarking to
#
# A number of parameters related to running the GPU code in HOOMD can be tuned for optimal performance.
# Unfortunately, the optimal values are impossible to predict and must be benchmarked. Additionally,
# the optimal values can vary due to different hardware and even compiler versions! HOOMD includes
# a balanced default set of tuning parameters benchmarked on various hardware configurations,
# all with the latest version of CUDA. 
#
# You might be able to boost the performance of your simulations over the default by a small amount
# if you run the tuning benchmark on your own machine. find_optimal_block_sizes() is the command
# that does this. In order to use it, run the following hoomd script:
#\code
#from hoomd_script import *
#tune.find_optimal_block_sizes()
#\endcode
#
# Be prepared to wait a while when running it. After it completes successfully, it will save the
# determined optimal tuning parameters in a file ~/.hoomd_block_tuning. This file will be 
# automatically read by any future invocations of HOOMD. 
#
# \note HOOMD ignores .hoomd_block_tuning files from older versions. You must rerun the tuning
# script after upgrading HOOMD. 
def find_optimal_block_sizes(save = True, only=None):
    util._disable_status_lines = True;

    # we cannot save if only is set
    if only:
        save = False;
    
    # list of force computes to tune
    fc_list = [ ('pair.table', 'pair_table_setup', 500),
                ('bond.bondtable', 'bond_table_setup', 2000),
                ('pair.lj', 'pair_lj_setup', 500),
                ('pair.slj', 'pair_slj_setup', 500),
                ('pair.yukawa', 'pair_yukawa_setup', 500),
                ('pair.ewald', 'pair_ewald_setup', 500),
                ('pair.cgcmm', 'pair_cgcmm_setup', 500),
                ('pair.gauss', 'pair_gauss_setup', 500),
                ('pair.morse', 'pair_morse_setup', 500),
                ('pair.dpd', 'pair_dpd_setup', 500),
                ('pair.dpdlj', 'pair_dpdlj_setup', 500),                
                ('pair.dpd_conservative', 'pair_dpd_conservative_setup', 500),
                ('bond.harmonic', 'bond.harmonic', 10000),
                ('angle.harmonic', 'angle.harmonic', 3000),
                ('angle.cgcmm', 'angle.cgcmm', 2000),
                ('dihedral.harmonic', 'dihedral.harmonic', 1000),
                ('improper.harmonic', 'improper.harmonic', 1000),
                ('bond.fene', 'bond_fene_setup', 2000)
                ];
    
    # setup the particle system to benchmark
    polymer = dict(bond_len=1.2, type=['A']*50, bond="linear", count=2000);
    N = len(polymer['type']) * polymer['count'];
    phi_p = 0.2;
    L = math.pow(math.pi * N / (6.0 * phi_p), 1.0/3.0);

    sysdef = init.create_random_polymers(box=hoomd.BoxDim(L), polymers=[polymer], separation=dict(A=0.35, B=0.35), seed=12)
    
    # need some angles, dihedrals, and impropers to benchmark
    angle_data = sysdef.sysdef.getAngleData();
    dihedral_data = sysdef.sysdef.getDihedralData();
    improper_data = sysdef.sysdef.getImproperData();
    num_particles = len(polymer['type']) * polymer['count'];
    
    for i in xrange(1,num_particles-3):
        angle_data.addAngle(hoomd.Angle(0, i, i+1, i+2));
    
    for i in xrange(1,num_particles-4):
        dihedral_data.addDihedral(hoomd.Dihedral(0, i, i+1, i+2, i+3));
        improper_data.addDihedral(hoomd.Dihedral(0, i, i+1, i+2, i+3));
    
    del angle_data
    del dihedral_data
    del improper_data
    del sysdef
    
    # run one time step to resort the particles for optimal memory access patterns
    hoomd_script.run(1);
    
    # list of optimal databases
    optimal_dbs = [];
    num_repeats = 3;
    for i in xrange(0,num_repeats):
        
        # initialize an empty database of optimal sizes
        optimal_db = {};
        
        # for each force compute
        for (fc_name,fc_init,n) in fc_list:
            if only and (not fc_name in only):
                continue

            globals.msg.notice(2, 'Benchmarking ' + str(fc_name) + '\n');
            # create it and benchmark it
            fc = eval(fc_init + '()')
            optimal = _find_optimal_block_size_fc(fc, n)
            optimal_db[fc_name] = optimal;
            
            # clean up
            fc.disable()
            del fc
        
        # now, benchmark the neighbor list
        if (only is None) or (only == 'nlist'):
            globals.msg.notice(2, 'Benchmarking nlist\n');
            lj = pair_lj_setup();
            optimal = _find_optimal_block_size_nl(globals.neighbor_list, 100)
            optimal_db['nlist'] = optimal;
            del lj;
        
        # and the neighbor list filtering
        if (only is None) or (only == 'nlist.filter'):
            globals.msg.notice(2, 'Benchmarking nlist.filter\n');
            lj = pair_lj_setup();
            globals.neighbor_list.reset_exclusions(exclusions = ['bond', 'angle'])
            optimal = _find_optimal_block_size_nl_filter(globals.neighbor_list, 200)
            optimal_db['nlist.filter'] = optimal;
            del lj;
        
        # add it to the list
        optimal_dbs.append(optimal_db);
    
    # print out all the optimal block sizes
    globals.msg.notice(2, '*****************\n');
    globals.msg.notice(2, 'Optimal block sizes found:\n');
    for db in optimal_dbs:
        globals.msg.notice(2, str(db) + '\n');
    
    # create a new db with the common optimal settings
    globals.msg.notice(2, "Chosing common optimal block sizes:\n");
    common_optimal_db = _choose_optimal_block_sizes(optimal_dbs);
    globals.msg.notice(2, str(common_optimal_db) + '\n');
        
    globals.msg.notice(2, '*****************\n')
    if save:
        _save_override_file(common_optimal_db);
    
    ## Currently not working for some reason.....
    # init.reset();
    util._disable_status_lines = False;

# functions for setting up potentials for benchmarking
def lj_table(r, rmin, rmax, epsilon, sigma):
    V = 4 * epsilon * ( (sigma / r)**12 - (sigma / r)**6);
    F = 4 * epsilon / r * ( 12 * (sigma / r)**12 - 6 * (sigma / r)**6);
    return (V, F)

def bond_table(r, rmin, rmax, kappa, r0):
    V = 0.5 * kappa * (r-r0)**2;
    F = -kappa*(r-r0);
    return (V, F)    

## \internal
# \brief Setup pair.table for benchmarking
def pair_table_setup():
    table = pair.table(width=1000);
    table.pair_coeff.set('A', 'A', func=lj_table, rmin=0.8, rmax=3.0, coeff=dict(epsilon=1.0, sigma=1.0));
    
    # no valid run() occurs, so we need to manually update the nlist
    globals.neighbor_list.update_rcut();
    return table;
    
## \internal
# \brief Setup pair.table for benchmarking
def bond_table_setup():
    btable = bond.bondtable(width=1000)
    btable.bond_coeff.set('polymer', func=bond_table, rmin=0.1, rmax=10.0, coeff=dict(kappa=330, r0=0.84))

    btable.update_coeffs();  
    return btable;    

## \internal
# \brief Setup pair.lj for benchmarking
def pair_lj_setup():
    fc = pair.lj(r_cut=3.0);
    fc.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0);
    
    # no valid run() occurs, so we need to manually update the nlist
    globals.neighbor_list.update_rcut();
    return fc;

## \internal
# \brief Setup pair.slj for benchmarking
def pair_slj_setup():
    fc = pair.slj(r_cut=3.0, d_max=1.0);
    fc.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0);
    
    # no valid run() occurs, so we need to manually update the nlist
    globals.neighbor_list.update_rcut();
    return fc;
    
## \internal
# \brief Setup pair.yukawa for benchmarking
def pair_yukawa_setup():
    fc = pair.yukawa(r_cut=3.0);
    fc.pair_coeff.set('A', 'A', epsilon=1.0, kappa=1.0);
    
    # no valid run() occurs, so we need to manually update the nlist
    globals.neighbor_list.update_rcut();
    return fc;

## \internal
# \brief Setup pair.ewald for benchmarking
def pair_ewald_setup():
    fc = pair.ewald(r_cut=3.0);
    fc.pair_coeff.set('A', 'A', kappa=1.0, grid=16, order=4);
    
    # no valid run() occurs, so we need to manually update the nlist
    globals.neighbor_list.update_rcut();
    return fc;

## \internal
# \brief Setup pair.cgcmm for benchmarking
def pair_cgcmm_setup():
    fc = pair.cgcmm(r_cut=3.0);
    fc.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0, exponents='LJ12-6');
    
    # no valid run() occurs, so we need to manually update the nlist
    globals.neighbor_list.update_rcut();
    return fc;

## \internal
# \brief Setup pair.cgcmm for benchmarking
def pair_gauss_setup():
    fc = pair.gauss(r_cut=3.0);
    fc.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0);
    
    # no valid run() occurs, so we need to manually update the nlist
    globals.neighbor_list.update_rcut();
    return fc;

## \internal
# \brief Setup pair.morse for benchmarking
def pair_morse_setup():
    fc = pair.morse(r_cut=3.0);
    fc.pair_coeff.set('A', 'A', D0=1.0, alpha=3.0, r0=1.0);
    
    # no valid run() occurs, so we need to manually update the nlist
    globals.neighbor_list.update_rcut();
    return fc;

## \internal
# \brief Setup pair.dpd for benchmarking
def pair_dpd_setup():
    fc = pair.dpd(r_cut=3.0, T=1.0);
    fc.pair_coeff.set('A', 'A', A=40.0, gamma=4.5); 
    
    # no valid run() occurs, so we need to manually update the nlist
    globals.neighbor_list.update_rcut();
    return fc;    

## \internal
# \brief Setup pair.dpdlj for benchmarking
def pair_dpdlj_setup():
    fc = pair.dpd(r_cut=3.0, T=1.0);
    fc.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, gamma=4.5); 
    
    # no valid run() occurs, so we need to manually update the nlist
    globals.neighbor_list.update_rcut();
    return fc;

## \internal
# \brief Setup pair.dpd_conservative for benchmarking
def pair_dpd_conservative_setup():
    fc = pair.dpd_conservative(r_cut=3.0);
    fc.pair_coeff.set('A', 'A', A=40);
    
    # no valid run() occurs, so we need to manually update the nlist
    globals.neighbor_list.update_rcut();
    return fc;

## \internal
# \brief Setup bond.fene for benchmarking
def bond_fene_setup():
    fc = bond.fene();
    fc.set_coeff('polymer', k=30.0, r0=3.0, sigma=1.0, epsilon=2.0);
    
    # no valid run() occurs, so we need to manually update the bond coeff
    fc.update_coeffs();
    return fc;

import hoomd;
import hoomd_script;
import init;
import globals;

## Make a series of short runs to determine the fastest performing r_buff setting
# \param warmup Number of time steps to run() to warm up the benchmark
# \param r_min Smallest value of r_buff to test
# \param r_max Largest value of r_buff to test
# \param jumps Number of different r_buff values to test
# \param steps Number of time steps to run() at each point
# \param set_max_check_period Set to True to enable automatic setting of the maximum nlist check_period
#
# tune.r_buff() executes \a warmup time steps. Then it sets the nlist \a r_buff value to \a r_min and runs for
# \a steps time steps. The TPS value is recorded, and the benchmark moves on to the next \a r_buff value
# completing at \a r_max in \a jumps jumps. Status information is printed out to the screen, and the optimal
# \a r_buff value is left set for further runs() to continue at optimal settings.
#
# Each benchmark is repeated 3 times and the median value chosen. Then, \a warmup time steps are run() again
# at the optimal r_buff in order to determine the maximum value of check_period. In total,
# (2*warmup + 3*jump*steps) time steps are run().
#
# \note By default, the maximum check_period is \b not set in tune.r_buff() for safety. If you wish to have it set
# when the call completes, call with the parameter set_max_check_period=True.
#
# \returns (optimal_r_buff, maximum check_period)
#
def r_buff(warmup=200000, r_min=0.05, r_max=1.0, jumps=20, steps=5000, set_max_check_period=False):
    # check if initialization has occurred
    if not init.is_initialized():
        globals.msg.error("Cannot tune r_buff before initialization\n");
    
    # check that there is a nlist
    if globals.neighbor_list is None:
        globals.msg.error("Cannot tune r_buff when there is no neighbor list\n");

    # start off at a check_period of 1
    globals.neighbor_list.set_params(check_period=1);

    # make the warmup run
    hoomd_script.run(warmup);
    
    # initialize scan variables
    dr = (r_max - r_min) / (jumps - 1);
    r_buff_list = [];
    tps_list = [];

    # loop over all desired r_buff points
    for i in xrange(0,jumps):
        # set the current r_buff
        r_buff = r_min + i * dr;
        globals.neighbor_list.set_params(r_buff=r_buff);
        
        # run the benchmark 3 times
        tps = [];
        hoomd_script.run(steps);
        tps.append(globals.system.getLastTPS())
        hoomd_script.run(steps);
        tps.append(globals.system.getLastTPS())
        hoomd_script.run(steps);
        tps.append(globals.system.getLastTPS())
        
        # record the median tps of the 3
        tps.sort();
        tps_list.append(tps[1]);
        r_buff_list.append(r_buff);
    
    # find the fastest r_buff
    fastest = tps_list.index(max(tps_list));
    fastest_r_buff = r_buff_list[fastest];

    # set the fastest and rerun the warmup steps to identify the max check period
    globals.neighbor_list.set_params(r_buff=fastest_r_buff);
    hoomd_script.run(warmup);

    # notify the user of the benchmark results
    globals.msg.notice(2, "r_buff = " + str(r_buff_list) + '\n');
    globals.msg.notice(2, "tps = " + str(tps_list) + '\n');
    globals.msg.notice(2, "Optimal r_buff: " + str(fastest_r_buff) + '\n');
    globals.msg.notice(2, "Maximum check_period: " + str(globals.neighbor_list.query_update_period()) + '\n');
    
    # set the found max check period
    if set_max_check_period:
        globals.neighbor_list.set_params(check_period=globals.neighbor_list.query_update_period());
    
    # return the results to the script
    return (fastest_r_buff, globals.neighbor_list.query_update_period());

