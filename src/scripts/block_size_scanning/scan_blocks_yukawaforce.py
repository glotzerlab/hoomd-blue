#! /usr/bin/python

import os;
import commands;
import re;
plotting = False;
try:
	import matplotlib.pyplot as plt;
	plotting = True;
except:
	print "matplotlib not found, disabling GUI plotting"

# base arguments for all runs
args_base = "-f Yukawa.GPU --half_nlist=0 -q --sort=1 -N 128000 --nsec=1";

bmark_time = [];
block_size_list = range(32,513,32);

for block_size in block_size_list:
	# run the benchmark and get the output
	bmark_output = commands.getoutput("./force_compute_bmark " + args_base + " --block_size " + str(block_size));
	if re.search("n/a s/step", bmark_output):
		print "probably reached maximum block size allowed, stopping"
		print bmark_output
		break;

	match = re.search("(\d*\.\d*) s/step", bmark_output);
	value = match.group(1);
	
	print block_size, value
	
	try:
		bmark_time.append(float(value));
	except ValueError:
		print "error parsing value"
		print bmark_output
		break
	

# block_size_list is likely larger than bmark_time, shrink them to the same size
block_size_list = block_size_list[0:len(bmark_time)];

# analyze the results and find the minimum
min_time = min(bmark_time);
min_idx = bmark_time.index(min_time);
max_time = max(bmark_time);
print "Fastest block is", max_time/min_time, "faster than the slowest";
print "Fastest block size is: ", block_size_list[min_idx];

# plot the results
if plotting:
	plt.plot(block_size_list, bmark_time, 's:')
	plt.xlabel('block size')
	plt.ylabel('time (ms)')
	plt.show()

