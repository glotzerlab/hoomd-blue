#!/usr/bin/python

from string import split

def process_column(lines, name, column):
	print name, " = {", ;

	i = 1;

	for line in lines:
		sl = split(line);
		value = sl[column];
		if (value[0] == '{'):
			value = value[1:-1];
		print value, ",",
		i+=1;
		if (i % 4) == 0:
			print "\n",;

	print "};"

file = open("nvt.tsv");
lines = file.readlines();

process_column(lines, "p", 1);
process_column(lines, "q", 2);
