#! /usr/bin/env hoomd

from hoomd_script import *
from hoomd_script import util
import time;
import random;
import numpy
import math

state = 'scanning'
num_scans = 3
buffer_width = 11
tps_buffers = numpy.zeros(shape=(num_scans, buffer_width))
cur_rbuff = 0.4
cur_buffer = 0;
cur_point = 0;
search_width = 0.4
rmin = cur_rbuff - search_width/2.0
rmax = cur_rbuff + search_width/2.0
r_points = numpy.linspace(rmin, rmax, buffer_width)
search_width_step = 0.1

run_steps = 10000
run_to = 0;

last_time = 0;
last_step = 0;

def find_max_perf():
    global search_width, r_points, rmin, rmax
    print tps_buffers
    tps_med = numpy.median(tps_buffers, axis=0)

    # find max value
    m = max(tps_med)
    mn = min(tps_med)
    diff = m/mn*100 - 100;
    print 'diff %', diff
    i0 = list(tps_med).index(m)
    max_rbuff = r_points[i0]

    # adjust search window
    if diff > 10:
        search_width -= search_width_step;
    
    if diff < 5:
        search_width += search_width_step;
    
    rmin = max(max_rbuff - search_width/2.0, 0)
    rmax = max_rbuff + search_width/2.0
    r_points = numpy.linspace(rmin, rmax, buffer_width)
    return max_rbuff;
    
def tune_nlist(step):
    global tps_buffers, state, cur_rbuff, search_width, rmin, rmax, last_time, last_step, cur_buffer, cur_point, run_to, r_points

    if state == 'scanning':
        # compute the current performance
        cur_time = time.time();
        cur_tps = float(step - last_step) / (cur_time - last_time);
        last_time = cur_time;
        last_step = step;
        
        tps_buffers[cur_buffer][cur_point] = cur_tps;
        cur_point += 1;
        
        if cur_point >= buffer_width:
            cur_point = 0;
            cur_buffer += 1;
        
        if cur_buffer >= num_scans:
            cur_buffer = 0;
            
            cur_rbuff = find_max_perf();
            
            # set the current buffer
            util._disable_status_lines = True;
            nlist.set_params(r_buff = cur_rbuff)
            print 'choosing', cur_rbuff;
            util._disable_status_lines = False
            state = 'running';
            run_to = step + run_steps;
        else:
            util._disable_status_lines = True;
            nlist.set_params(r_buff = r_points[cur_point])
            print 'searching', r_points[cur_point];
            util._disable_status_lines = False
        
    if state == 'running':
        if step >= run_to:
            # set the search buffer
            util._disable_status_lines = True;
            nlist.set_params(r_buff = r_points[cur_point])
            print 'searching cur_buff';
            util._disable_status_lines = False
            
            state = 'scanning';
            r_points = numpy.linspace(rmin, rmax, buffer_width)
    
init.create_random(N=10000, phi_p=0.3)
lj = pair.lj(r_cut=3.0)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0)
nlist.set_params(r_buff = 0.65)

all = group.all()
integrate.mode_standard(dt=0.005)
integrate.nvt(group=all, T=3.0, tau=0.5)

run(4000)
run(130000, callback_period=1000, callback=tune_nlist)
run(130000, callback_period=1000, callback=tune_nlist)
