#! /usr/bin/python

# This script converts a hoomd xml file to a LAMMPS dump file
import xml.dom.minidom
import sys
from optparse import OptionParser

parser = OptionParser(usage="hoomd2lammps_dump.py -o atoms.dump input0.xml [input 1.xml ...]")
parser.add_option("-o", dest="outfile", help="LAMMPS dump file to write")
parser.add_option("--noscale", dest="scale", action="store_false", default=True, help="Disable scaled coordinates")

(options, args) = parser.parse_args()

if not options.outfile:
	parser.error("You must specify an output file");

if len(args) == 0:
	parser.error("You must specify at least one input file");

# open the output file
f = file(options.outfile, 'w');

for fname in args:
	print "Reading", fname
	dom = xml.dom.minidom.parse(fname);
	
	# start by parsing the file
	hoomd_xml = dom.getElementsByTagName('hoomd_xml')[0];
	configuration = hoomd_xml.getElementsByTagName('configuration')[0];
	
	if configuration.hasAttribute('time_step'):
		time_step = int(configuration.getAttribute('time_step'));
	else:
		time_step = 0;
	
	# read the box size
	box 	= configuration.getElementsByTagName('box')[0];
	Lx = box.getAttribute('Lx');
	Ly = box.getAttribute('Ly');
	Lz = box.getAttribute('Lz');
	
	# parse the particle coordinates
	position = configuration.getElementsByTagName('position')[0];
	position_text = position.childNodes[0].data
	xyz = position_text.split()
	print "Found", len(xyz)/3, " particles";
	
	# parse the velocities
	velocity_nodes = configuration.getElementsByTagName('velocity')
	velocity_xyz = [];
	if len(velocity_nodes) == 1:
		velocity = velocity_nodes[0];
		velocity_text = velocity.childNodes[0].data
		velocity_xyz = velocity_text.split()
		if len(velocity_xyz) != len(xyz):
			print "Error Number of velocities doesn't match the number of positions"
			sys.exit(1);
		print "Found", len(velocity_xyz)/3, " velocities";
	
	# parse the particle types
	type_nodes = configuration.getElementsByTagName('type');
	if len(type_nodes) == 1:
		type_text = type_nodes[0].childNodes[0].data;
		type_names = type_text.split();
		if len(type_names) != len(xyz)/3:
			print "Error! Number of types differes from the number of particles"
			sys.exit(1);
	else:
		print "Error! The type node must be in the xml file"
		sys.exit(1);
	
	# convert type names to type ids
	type_id = [];
	type_id_mapping = {};
	for name in type_names:
		name = name.encode();
		# use the exising mapping if we have made one
		if name in type_id_mapping:
			type_id.append(type_id_mapping[name]);
		else:
			# otherwise, we need to create a new mapping
			type_id_mapping[name] = len(type_id_mapping)+1;
			type_id.append(type_id_mapping[name]);
	
	print "Mapped particle types:"
	print type_id_mapping
	
	# parse the bonds
	bond_nodes = configuration.getElementsByTagName('bond')
	bond_a = [];
	bond_b = [];
	bond_type_id = [];
	bond_type_id_mapping = {};
	if len(bond_nodes) == 1:
		bond = bond_nodes[0];
		bond_text = bond.childNodes[0].data.encode();
		bond_raw = bond_text.split();
		
		# loop through the bonds and split the a,b and type from the raw stream
		# map types names to numbers along the way
		for i in xrange(0,len(bond_raw),3):
			bond_a.append(bond_raw[i+1]);
			bond_b.append(bond_raw[i+2]);
			
			# use the exising mapping if we have made one
			name = bond_raw[i];
			if name in bond_type_id_mapping:
				bond_type_id.append(bond_type_id_mapping[name]);
			else:
				# otherwise, we need to create a new mapping
				bond_type_id_mapping[name] = len(bond_type_id_mapping)+1;
				bond_type_id.append(bond_type_id_mapping[name]);
				
		print "Found", len(bond_a), "bonds";
		print "Mapped bond types:"
		print bond_type_id_mapping;
		
	# now we have everything and can write the time step to the LAMMPS output file
	f.write("ITEM: TIMESTEP\n%d\n" % (time_step));
	f.write("ITEM: NUMBER OF ATOMS\n%d\n" % (len(xyz)/3));
	
	f.write("ITEM: BOX BOUNDS\n");
	f.write("%f %f\n" % (-float(Lx)/2.0, float(Lx)/2.0));
	f.write("%f %f\n" % (-float(Ly)/2.0, float(Ly)/2.0));
	f.write("%f %f\n" % (-float(Lz)/2.0, float(Lz)/2.0));
	
	f.write("ITEM: ATOMS\n");
	for i in xrange(0,len(xyz)/3):
		x = float(xyz[i*3]);
		y = float(xyz[i*3+1]);
		z = float(xyz[i*3+2]);
		# map the x,y,z coords to scaled box coordinates if requested by the user
		if options.scale:
			x = (x + float(Lx)/2.0) / float(Lx);
			y = (y + float(Ly)/2.0) / float(Ly);
			z = (z + float(Lz)/2.0) / float(Lz);
		# write the line to the file
		f.write("%d %d %f %f %f\n" % (i+1, type_id[i], x, y, z));
		
f.close()
