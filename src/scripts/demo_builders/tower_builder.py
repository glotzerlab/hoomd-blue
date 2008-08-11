#! /usr/bin/python

## generated particle positions
class position:
	def __init__(self,type, x,y,z):
		self.type = type;
		self.x = x;
		self.y = y;
		self.z = z;

positions = [];

## generated bonds
class bond:
	def __init__(self, type, a, b):
		self.type = type;
		self.a = a;
		self.b = b;

bonds = [];

class wall:
	def __init__(self, ox,oy,oz,nx,ny,nz):
		self.ox = ox;
		self.oy = oy;
		self.oz = oz;
		self.nx = nx;
		self.ny = ny;
		self.nz = nz;

foundation_walls = [];

## creates a single box
# \param x coordinate of the lower left particle
# \param y coordinate of the lower left particle
# \param z coordinate of the lower left particle
# \param L length of one side of the box
def create_box(x, y, z, size, type):
	# track the current particle index
	idx = len(positions);
	# add the 8 particles
	positions.append(position(type, x, y, z));					# idx
	positions.append(position(type, x, y, z+size));				# idx+1
	positions.append(position(type, x, y+size, z));				# idx+2
	positions.append(position(type, x, y+size, z+size));			# idx+3
	positions.append(position(type, x+size, y, z));				# idx+4
	positions.append(position(type, x+size, y, z+size));			# idx+5
	positions.append(position(type, x+size, y+size, z));			# idx+6
	positions.append(position(type, x+size, y+size, z+size));		# idx+7

	# add the bonds
	bonds.append(bond('block', idx, idx+1))
	bonds.append(bond('block', idx, idx+2))
	bonds.append(bond('block', idx, idx+4))
	bonds.append(bond('block', idx+1, idx+3))
	bonds.append(bond('block', idx+1, idx+5))
	bonds.append(bond('block', idx+2, idx+3))
	bonds.append(bond('block', idx+2, idx+6))
	bonds.append(bond('block', idx+4, idx+6))
	bonds.append(bond('block', idx+4, idx+5))
	bonds.append(bond('block', idx+3, idx+7))
	bonds.append(bond('block', idx+5, idx+7))
	bonds.append(bond('block', idx+6, idx+7))
	
	# cross braces
	bonds.append(bond('cross', idx+0, idx+7))
	bonds.append(bond('cross', idx+3, idx+4))
	bonds.append(bond('cross', idx+2, idx+5))
	bonds.append(bond('cross', idx+6, idx+1))

## create a rectangular tower of blocks
# \param x lower left x coordinate
# \param y lower left y coordinate
# \param z lower left z coordinate
# \param size size of each block
# \param width width of the tower in blocks
# \param height height of the tower in blocks
# \param depth depth of the tower in blocks
def create_tower(x,y,z,size, width, height, depth):
	# create the base
	for cur_i in xrange(0,width):
		for cur_k in xrange(0,depth):
			create_box(x + (2*cur_i)*size, y, z + (2*cur_k)*size, size, 'N');

	# add the walls to hold the foundataion
	foundation_walls.append(wall(x-1.12246, 0, 0, 1, 0, 0));
	foundation_walls.append(wall(x + 2*(width-0.5)+1.12246, 0, 0, -1, 0, 0));
	foundation_walls.append(wall(0, 0, z + 2*(depth-0.5)+1.12246, 0, 0, -1));
	foundation_walls.append(wall(0, 0, z-1.12246, 0, 0, 1));
			
	for cur_j in xrange(1,height):
		#offset = 0.5 * (cur_j % 2);
		offset = 0;
		for cur_i in xrange(0,width):
			for cur_k in xrange(0,depth):
				create_box(x + (2*cur_i+offset)*size, y + (1.9*cur_j)*size, z + (2*cur_k+offset)*size, size, 'A');

## create a pyramid of blocks
# \param x lower left x coordinate
# \param y lower left y coordinate
# \param z lower left z coordinate
# \param size size of each block
# \param height height of the tower in blocks
def create_pyramid(x,y,z,size,height):
	# create the base
	for cur_i in xrange(0,height):
		for cur_k in xrange(0,height):
			positions.append(position('N', x + cur_i*size, y, z + cur_k*size));

	# add the walls to hold the foundataion
	foundation_walls.append(wall(x-1.12246, 0, 0, 1, 0, 0));
	foundation_walls.append(wall(x + height*size-1.0+1.12246, 0, 0, -1, 0, 0));
	foundation_walls.append(wall(0, 0, z + height*size-1.0+1.12246, 0, 0, -1));
	foundation_walls.append(wall(0, 0, z-1.12246, 0, 0, 1));
			
	for cur_j in xrange(1,height):
		offset = cur_j*0.5;
		for cur_i in xrange(0,height-cur_j):
			for cur_k in xrange(0,height-cur_j):
				positions.append(position('A', x + (cur_i+offset)*size, y + 0.707107*cur_j*size, z + (cur_k+offset)*size));

## write the xml file to disk
# \param fname file name to write
def write_xml(fname, Lx, Ly, Lz):
	f = file(fname, 'w');
	f.write('<?xml version ="1.0" encoding ="UTF-8" ?>\n');
	f.write('<hoomd_xml>\n');
	f.write('<configuration time_step="0">\n');
	f.write('<box Units ="sigma"  Lx="%f" Ly= "%f" Lz="%f" />\n' % (Lx, Ly, Lz));
	
	f.write('<position units ="sigma" >\n');
	for position in positions:
		f.write('%f %f %f\n' % (position.x, position.y, position.z));
	f.write('</position>\n');
	
	f.write('<type>\n');
	for position in positions:
		f.write('%s\n' % (position.type));
	f.write('</type>\n');

	if len(bonds) > 1:
		f.write('<bond>\n');
		for bond in bonds:
			f.write('%s %d %d\n' % (bond.type, bond.a, bond.b));
		f.write('</bond>\n');

	f.write('<wall>\n');
	f.write('<coord ox="0" oy="%f" oz="0" nx="0" ny="-1" nz="0" />\n' % (Ly/2.0 - 1.0));
	f.write('<coord ox="0" oy="%f" oz="0" nx="0" ny="1" nz="0" />\n' % (-Ly/2.0 + 1.0));
	f.write('<coord ox="%f" oy="0" oz="0" nx="1" ny="0" nz="0" />\n' % (-Lx/2.0 + 1.0));
	f.write('<coord ox="%f" oy="0" oz="0" nx="-1" ny="0" nz="0" />\n' % (Lx/2.0 - 1.0));
	f.write('<coord ox="0" oy="0" oz="%f" nx="0" ny="" nz="-1" />\n' % (-Lz/2.0 + 1.0));
	f.write('<coord ox="0" oy="0" oz="%f" nx="0" ny="0" nz="1" />\n' % (Lz/2.0 - 1.0));
	for wall in foundation_walls:
		f.write('<coord ox="%f" oy="%f" oz="%f" nx="%f" ny="%f" nz="%f" />\n' % (wall.ox, wall.oy, wall.oz, wall.nx, wall.ny, wall.nz));
	f.write('</wall>\n');

	f.write('</configuration>\n');
	f.write('</hoomd_xml>\n');

create_pyramid(0, -23, 0, 1.0, 5);
write_xml('test.xml', 50, 50, 50);
