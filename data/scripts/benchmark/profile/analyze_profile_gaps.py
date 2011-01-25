#! /usr/bin/python

import csv
import numpy
import sys
from subprocess import *

if len(sys.argv) < 2:
    print "Usage: analyze_profile_gaps.py filename.csv [methodname called once per step";

mark_method = None;
if len(sys.argv) == 3:
    mark_method = sys.argv[2];

# store marks at each time step
marks = [];

# store white space prior to each kernel call
idletime = {};

# previous kernel call info for delta calculations
prev_timestamp = 0;
prev_gputime = 0;

reader = csv.reader(open(sys.argv[1], "rb"), delimiter=',')
discard = 0;
for row in reader:
    if row[0][0] == '#':
        continue;

    if row[0] == 'gpustarttimestamp':
        continue;
    
    # get the current timestamp, method name, and gputime
    timestamp = int(row[0], 16);
    gputime = float(row[2]);
    method = row[1];

    # discard the first 2000 lines
    if discard < 2000:
        discard += 1;
        prev_timestamp = timestamp;
        prev_gputime = gputime;
        continue;

    # if this is the mark method, record the mark time
    if method == mark_method:
        marks.append(timestamp);
    
    idle = float(timestamp - prev_timestamp)/1e3 - prev_gputime;
    
    if method in idletime:
        idletime[method].append(idle);
    else:
        idletime[method] = [idle];
    
    prev_timestamp = timestamp;
    prev_gputime = gputime;


# analyze the marks to determine average and median TPS
# first, throw out the first 1000 marks, then compute the differences between the remaining ones
if mark_method is not None and len(marks):
    tstep_size = [];
    for i in xrange(len(marks)-1):
        tstep_size.append(marks[i+1] - marks[i]);
    
    print "Average us/step : %3.2f" % (float(numpy.average(tstep_size))/1e3);
    print " Median us/step : %3.2f" % (float(numpy.median(tstep_size))/1e3);
    print

for k,v in idletime.iteritems():
    pretty_name = Popen(["c++filt", "-p", k], stdout=PIPE).communicate()[0].strip()
    print pretty_name.rjust(60), ":", numpy.median(v)


