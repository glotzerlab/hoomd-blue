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

# fit it to a parabola
A = numpy.vander(r_data, 3)
(coeffs, residuals, rank, sing_vals) = numpy.linalg.lstsq(A, tps_data)
f = numpy.poly1d(coeffs)
r_est = numpy.linspace(0.3,1.3,200)
tps_est = f(r_est)

m = max(tps_est)
i = list(tps_est).index(m)
print 'rmax_parab', r_est[i]

# find max value
m = max(tps_data[0:10])
mn = min(tps_data[0:10])
print 'diff %', m/mn*100 - 100
i0 = list(tps_data[0:10]).index(m)
print 'rmax', r_data[i0]

m = max(tps_data[11:21])
mn = min(tps_data[11:21])
print 'diff %', m/mn*100 - 100
i1 = list(tps_data[11:21]).index(m)
print 'rmax', r_data[i1]

m = max(tps_data[22:32])
mn = min(tps_data[22:32])
print 'diff %', m/mn*100 - 100
i2 = list(tps_data[22:32]).index(m)
print 'rmax', r_data[i2]

print 'rmax_avg', (r_data[i0] + r_data[i1] + r_data[i2])/3

fig = pyplot.figure();
pyplot.plot(r_data[0:10], tps_data[0:10], r_data[11:21], tps_data[11:21], r_data[22:32], tps_data[22:32])

pyplot.show()

