use strict;
use warnings;
use File::Find;
use Text::Diff::Parser;
use File::Temp;

# parameters for controlling what gets reported
my $max_line_len = 121;
my $overlength_threshold = 25;
my $astyle_changes_threshold = 25;

my $skip_svn_processing = 0;

# format with astyle and count the number of lines that it changed
sub process_file_astyle
    {
    my ($fname, $lines) = @_;
    $_ = $fname;

    my $message = "";

    if (/\.cc$/ or /\.h$/ or /\.cu$/ or /\.cuh$/)
        {
        # get a temporary file to write to
        my $astyle_out_file = tmpnam();
        # restyle the file
        `astyle --mode=c --style=whitesmith --keep-one-line-statements --convert-tabs  --fill-empty-lines -M70 < $fname > $astyle_out_file`;

        # get the differences between the files
        my $diffs = `diff -u $fname $astyle_out_file`;

        my $parser = Text::Diff::Parser->new($diffs);
        $parser->simplify();
        my $total_changes = 0;
        foreach my $change ( $parser->changes ) 
                {
                $total_changes += $change->size;
                }

        # output the number of changes
        if ($total_changes > $astyle_changes_threshold)
            {
            $message .= "astyle changes: $total_changes\n";
            }

        unlink($astyle_out_file);
        }

    return $message;
    }

# check that the file has the proper svn properties set
sub process_file_svn
    {
    my $fname = $_[0];
    
    my $message = "";

    if ($skip_svn_processing)
        {
        return $message;
        }
    
    # check keywords
    my $keywords = `svn propget svn:keywords $fname`;
    chomp $keywords;
    if (not ($keywords =~ /Id/ and $keywords =~ /URL/))
        {
        $message .= "invalid svn:keywords:  $keywords\n";
        }

    # check eol-style
    my $eolstyle = `svn propget svn:eol-style $fname`;
    chomp $eolstyle;
    if (not ($eolstyle =~ /^native$/))
        {
        $message .= "invalid svn:eol-style: $eolstyle\n";
        }

    return $message;
    }

# process all the lines in the file and check for tabs, eof newline, overlength conditions, etc.
sub process_file_lines
    {
    my $fname = $_[0];
    # for checking the final newline
    my $last_line;
    
    # open the file
    open(FILE, "< $fname") or die "can't open $fname: $!";
    
    # initialize counters to 0
    my $tab_count = 0;
    my $line_count = 0;
    my $overlength_count = 0;
    # loop through all lines in the file and add up counters
    while (<FILE>)
        {
        $tab_count += tr/\t//;
        $last_line = $_ if eof;

        if (length($_) > $max_line_len)
            {
            $overlength_count += 1;
            }

        $line_count += 1;
        }
    close(FILE);

    my $message = "";

    if ($tab_count > 0)
        {
        $message .= "tabs:                $tab_count\n";
        }
    if ($overlength_count > $overlength_threshold)
        {
        $message .= "lines overlength:    $overlength_count\n";
        }
    if (not $last_line =~ /^\n/)
        {
        $message .= "end of file newline: missing\n";
        }

    return ($message, $line_count);
    }

sub wanted 
    {
    my $fname = $_;    
    
    # skip processing if this file is in the extern directory
    if ($File::Find::name =~ /\/extern\//)
        {
        return;
        }
    
    # skip processing if this file is in the share directory
    if ($File::Find::name =~ /\/share\//)
        {
        return;
        }
    
    if (/\.cc$/ or /\.h$/ or /\.cu$/ or /\.cuh$/ or /\.py$/)
        {
        my $full_message = "";
        my $message;
        my $line_count;
        ($message, $line_count) = process_file_lines($fname);
        $full_message .= $message;
        $full_message .= process_file_svn($fname);
        #$full_message .= process_file_astyle($fname, $line_count);

        if ($full_message)
            {
            print "$File::Find::name\n";
            print $full_message;
            print "\n";
            }
        }
    }

# turn off svn processing if this is not in a svn checkout
my $svnversion = `svnversion`;
if ($svnversion =~ /exported/)
    {
    print "Not a svn checkout, skipping svn properties checks\n\n";
    $skip_svn_processing = 1;
    }

# grep through the source and look for problems
finddepth(\&wanted, '.');
