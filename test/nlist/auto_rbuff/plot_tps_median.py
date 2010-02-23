import numpy
import scipy
import csv
from matplotlib import pyplot

data = csv.DictReader(open('lj_rbuff_tps'), delimiter="\t")

# read in data output from hoomd
r_data = [];
tps_data = [];

for line in data:
    r_data.append(float(line['r']))
    tps_data.append(float(line['TPS']))

r_med = r_data[0:10];
tmp = numpy.array([tps_data[0:10], tps_data[11:21], tps_data[22:32]])
tps_med = numpy.median(tmp, axis=0)

# find max value
m = max(tps_med)
mn = min(tps_med)
print 'diff %', m/mn*100 - 100
i0 = list(tps_med).index(m)
print 'rmax', r_med[i0]

fig = pyplot.figure();
pyplot.plot(r_med, tps_med, r_data[0:10], tps_data[0:10], r_data[11:21], tps_data[11:21], r_data[22:32], tps_data[22:32])

pyplot.show()
