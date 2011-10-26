#! /usr/bin/env perl

open(IN, "<$ENV{HOME}/Documents/Papers/bib/HOOMD-doc.bib") or die("No input file");
open(OUT, '>doc/user/HOOMD.bib');

$waiting = 0;
while (<IN>)
    {
    if (/^file/ || /^mendeley-tags/ || /^abstract/ || /^keywords/ || /^url/)
        {
        $waiting = 1;
        }
    if ($waiting)
        {
        if (/\}/)
            {
            $waiting = 0;
            next;
            }
        else
            {
            next;
            }
        }
    print OUT;
    }

