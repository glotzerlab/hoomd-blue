from __future__ import print_function
import sys
from subprocess import call
import os

if len(sys.argv) == 1:
    print("invalid input");
    sys.exit(1)

if sys.argv[1] != 'install':
    print("invalid input");
    sys.exit(1)

if len(sys.argv) == 3 and sys.argv[2] != '--force':
    print("invalid input");
    sys.exit(1)

if len(sys.argv) > 3:
    print("invalid input");
    sys.exit(1)

os.environ['CC'] = '/usr/bin/clang'
os.environ['CXX'] = '/usr/bin/clang++'

call(["mkdir", "-p", "build"])
os.chdir("build");
call(["cmake", "../", "-DINSTALL_SITE=on", "-DBUILD_TESTING=off"])
call(["make", "install"])
