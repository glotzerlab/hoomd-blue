#! /usr/bin/python

# This script converts a set of raw coordinates to a hoomd xml file
import sys
import math
from optparse import OptionParser

parser = OptionParser(usage="raw2hoomd.py -o hoomd_init.xml input.txt")
parser.add_option("-o", dest="outfile", help="HOOMD XML file to write")
parser.add_option("--dim", dest="dimension", help="Specify dimension 1, 2 or 3, (default 3)")
parser.add_option("--box", dest="box", help="Specify lx,ly,lz, Default is a bounding box")
parser.add_option("--type", dest="type", help="Particle type (default A)")

(options, args) = parser.parse_args()

if not options.outfile:
    parser.error("You must specify an output file");
    
# Does NOT specify the dimension of the simulation, just the raw coordinates    
if not options.dimension:
    dim = 3
else:
    dim = int(options.dimension)
    if dim > 3 or dim < 1:
        parser.error("Dimension specified out of range");

if len(args) == 0:
    parser.error("You must specify at least one input file");

# open the output file
f = file(options.outfile, 'w');

for fname in args:
    print "Reading", fname
    
    coordfile = open(fname,'r');
    
    f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
    f.write("<hoomd_xml version=\"1.2\">\n")
    f.write("<!-- positions derived from raw coordinates file ")
    f.write(fname)
    f.write(" -->\n")
    f.write("<configuration time_step=\"0\">\n")
    f.write("<position>\n")

    pcount = 0;
    maxx = 0;
    minx = 0;
    maxy = 0;
    miny = 0;
    maxz = 0;
    minz = 0;
    
    floatcount = 0;
    for line in coordfile:
    # Parse the line into floats
        for val in line.split():
            fv = float(val);
        
            if (floatcount ==0):
                if fv > maxx: maxx = fv;
                if fv < minx: minx = fv;
                
            if (floatcount ==1):
                if fv > maxy: maxy = fv;
                if fv < miny: miny = fv;
   
            if (floatcount ==2):
                if fv > maxz: maxz = fv;
                if fv < minz: minz = fv;                             
        
            if floatcount < dim - 1:
                f.write("%f "% fv)                                
                floatcount = floatcount+1;
            else:
                f.write("%f"% fv)
                for i in xrange(0,3-dim):
                    f.write(" 0.0")
                f.write("\n");
                floatcount = 0;
                pcount = pcount + 1;
                
    if floatcount != 0:
        raise ValueError("Wrong number of coordinates in file\n");
        
    f.write("</position>\n")
    
    # Determine Simulation Box
    if not options.box:
        lx = max(2*(maxx-minx + (maxx + minx)/2), 10)
        ly = max(2*(maxy-miny + (maxy + miny)/2), 10)
        lz = max(2*(maxz-minz + (maxz + minz)/2), 10)
    else :
        boxvals = options.box.split(',')
        lx = float(boxvals[0])
        ly = float(boxvals[1])
        lz = float(boxvals[2])
    
    f.write("<box lx=\"%f\" ly=\"%f\" lz=\"%f\"/>\n"%(lx,ly,lz));    

    # Write out Type
    f.write("<type>\n")
    
    if not options.type:
        ptype = 'A'
    else:
        ptype = options.type
        
    for i in xrange(0,pcount): 
        f.write("%s\n"%ptype);
        
    f.write("</type>\n")

    # End File
    f.write("</configuration>\n")
    f.write("</hoomd_xml>\n")
f.close()

