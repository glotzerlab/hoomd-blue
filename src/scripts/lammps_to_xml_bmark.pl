#!/usr/bin/perl -w

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


# this particular perl script is written soley to take in the lammps
# data files in benchmarks/data and write out hoomd xml files

sub print_usage
    {
    # Print usage
    print "Usage:\n";
    print "COM_rearrange.pl indata.m output\n";
    exit;
    }

# Parse arguments
if (@ARGV != 2)
    {
    print_usage();
    exit;
    }

my $infile = $ARGV[0];
my $output_file = $ARGV[1];

open(OUTFILE, ">$output_file") or die("Cannot open $output_file for reading");
open(INFILE, $infile) or die("Cannot open $infile for writing");

# read in the header line
my $line = <INFILE>;
$line = <INFILE>;

# read in the number of atoms and bonds
$line = <INFILE>;
chomp($line);
my @split_line = split(/\s+/, $line);
my $N_particles = $split_line[0];
print "Reading $N_particles particles\n";
$line = <INFILE>;
chomp($line);
@split_line = split(/\s+/, $line);
my $N_bonds = $split_line[0];
print "Reading $N_bonds bonds\n";

# read in the number of atom types
$line = <INFILE>;
$line = <INFILE>;
chomp($line);
@split_line = split(/\s+/, $line);
my $N_types = $split_line[0];

# skip the number of bond types
$line = <INFILE>;
$line = <INFILE>;

# read in the box (assume cubic)
$line = <INFILE>;
$line = <INFILE>;
$line = <INFILE>;
@split_line = split(/\s+/, $line);
my $box_width = $split_line[1] - $split_line[0];

# read in the masses
$line = <INFILE>;
$line = <INFILE>;
$line = <INFILE>;
$line = <INFILE>;
$line = <INFILE>;
$line = <INFILE>;

# expect atoms
$line = <INFILE>;
chomp($line);
if ($line ne "Atoms")
	{
	die("Expected Atoms");
	}
$line = <INFILE>;

# read in all the atoms
# start by preallocating an array
my @x;
my @y;
my @z;
my @type;
for (my $i = 0; $i < $N_particles; $i++)
	{
	push(@x, 0.0);
	push(@y, 0.0);
	push(@z, 0.0);
	push(@type, 0);
	}

# now read in the atoms
for (my $i = 0; $i < $N_particles; $i++)
	{
	$line = <INFILE>;
	@split_line = split(/\s+/, $line);
	my $tag = $split_line[0]-1;
	$x[$tag] = $split_line[3];
	$y[$tag] = $split_line[4];
	$z[$tag] = $split_line[5];
	$type[$tag] = $split_line[2]-1;
	}

# read in the velocities the same way
# expect Velocities
# $line = <INFILE>;
# $line = <INFILE>;
# chomp($line);
# if ($line ne "Velocities")
# 	{
# 	die("Expected Velocities");
# 	}
# $line = <INFILE>;
# 
# # start by preallocating an array
# my @vx;
# my @vy;
# my @vz;
# for (my $i = 0; $i < $N_particles; $i++)
# 	{
# 	push(@vx, 0.0);
# 	push(@vy, 0.0);
# 	push(@vz, 0.0);
# 	}
# 
# # now read in the velocities
# for (my $i = 0; $i < $N_particles; $i++)
# 	{
# 	$line = <INFILE>;
# 	@split_line = split(/\s+/, $line);
# 	my $tag = $split_line[0]-1;
# 	$vx[$tag] = $split_line[1];
# 	$vy[$tag] = $split_line[2];
# 	$vz[$tag] = $split_line[3];
# 	}

# time to read in the bonds now
$line = <INFILE>;
$line = <INFILE>;
chomp($line);
if ($line ne "Bonds")
	{
	die("Expected Bonds");
	}
$line = <INFILE>;

# read in the bonds the same way
my @bond_a;
my @bond_b;
for (my $i = 0; $i < $N_bonds; $i++)
	{
	$line = <INFILE>;
	@split_line = split(/\s+/, $line);
	push(@bond_a, $split_line[2]-1);
	push(@bond_b, $split_line[3]-1);
	}

close(INFILE);

# time to write the output file
print OUTFILE "<?xml version =\"1.0\" encoding =\"UTF-8\" ?>\n";
print OUTFILE "<HOOMD_xml>\n";
print OUTFILE "<Configuration time_step=\"0\" N=\"$N_particles\" NTypes =\"$N_types\" NBonds =\"$N_bonds\" />\n";
print OUTFILE "<Box Units =\"sigma\"  Lx=\"$box_width\" Ly= \"$box_width\" Lz=\"$box_width\" />\n";
print OUTFILE "<Position units =\"sigma\" >\n";

for (my $i = 0; $i < $N_particles; $i++)
	{
	print OUTFILE "$x[$i] $y[$i] $z[$i]\n";
	}
print OUTFILE "</Position>";

# print OUTFILE "<Velocity units =\"sigma/tau\">\n";
# for (my $i = 0; $i < $N_particles; $i++)
# 	{
# 	print OUTFILE "$vx[$i] $vy[$i] $vz[$i]\n";
# 	}
# print OUTFILE "</Velocity>\n";
print OUTFILE "<Type>\n";

for (my $i = 0; $i < $N_particles; $i++)
	{
	print OUTFILE "$type[$i]\n";
	}
print OUTFILE "</Type>\n";

print OUTFILE "<Bonds>\n";
for (my $i = 0; $i < $N_bonds; $i++)
	{
	print OUTFILE "$bond_a[$i] $bond_b[$i]\n";
	}
print OUTFILE "</Bonds>\n";
print OUTFILE "</HOOMD_xml>\n";

close(OUTFILE);
