use File::Find;

@file_list = ();

sub wanted 
    {
    # skip processing if this file is in the extern directory
    if ($File::Find::name =~ /\/extern\//)
        {
        return;
        }

    # skip processing if this file is in the build directory
    if ($File::Find::name =~ /\/build\//)
        {
        return;
        }
    
    # skip processing if this file is in the share directory
    if ($File::Find::name =~ /\/share\//)
        {
        return;
        }
    
    # skip processing if this file is in the test directory
    if ($File::Find::name =~ /\/test\//)
        {
        return;
        }
    
    # skip processing if this file is in the data directory
    if ($File::Find::name =~ /\/data\//)
        {
        return;
        }

    # skip processing if this file is in the microbenchmarks directory
    if ($File::Find::name =~ /\/microbenchmarks\//)
        {
        return;
        }    

    if (/\.cc$/ or /\.h$/ or /\.cu$/ or /\.cuh$/ or /\.py$/ or /\.txt$/)
        {
        open(FILE, "< $_") or die "can't open $_: $!";
        
        $found = 0;
        while (<FILE>)
            {
            chomp;
            if (/[\/#\s]*Maintainer: ([\w]*)[\s]*[\/]*[\s]*(.*)$/)
                {
                # print "||\`$File::Find::name\`||$1||$2||\n";
                push @file_list, {fname => "$File::Find::name", user => "$1", notes => "$2"};
                $found = 1;
                last;
                }
            }

        if (not $found)
            {
            # print "||$File::Find::name||''none''||||\n";
            push @file_list, {fname => "$File::Find::name", user => "''none''", notes => ""};
            }
        close(FILE);
        }
    }

# header
print "|._File|._Maintainer|._Notes|\n";

# grep through the source
finddepth(\&wanted, '.');

@ordered_file_list = sort { $a->{user} cmp $b->{user} || $a->{fname} cmp $b->{fname} } @file_list;

for $file ( @ordered_file_list )
    {
    print "|\@$file->{fname}\@|$file->{user}|$file->{notes}|\n";
    }
