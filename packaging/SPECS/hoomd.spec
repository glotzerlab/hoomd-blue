%global	repository	https://codeblue.umich.edu/git/hoomd-blue
# The git refspec to build is specified by defining %{refspec}.
# The version number to tag the build with is %{version}.
# Both should be specified, as we cannot determine the version until we have checked out the code
#  - if they are not speicified, then the most recent tag is built
%global version	%{?version}%{!?version:0.11.3}
%global refspec	%{?refspec}%{!?refspec:v0.11.3}
%global release	%{?release}%{!?release:0}

# the Red Hat convention is to put 64-bit libs in lib64
%global libsuffix	%(uname -p |sed -n 's/.*64$/64/p')
%global libname		lib%(uname -p |sed -n 's/.*64$/64/p')
%global python		%(which python)
%global sitedir		%(%{python} -c "from distutils.sysconfig import get_python_lib; print get_python_lib(plat_specific=True)")

BuildRoot:		%{_tmppath}/%{name}-root
Summary: 		Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
License: 		various
Name: 			hoomd
Version: 		%{version}
Release: 		%{release}
# sources will be retrieved with subversion
# Source: 		http://codeblue.umich.edu/hoomd-blue/downloads/0.11/hoomd-0.11.3.tar.bz2
URL:			http://codeblue.umich.edu/hoomd-blue/
Prefix:			/usr
Group: 			Applications
BuildRequires:		gcc-c++, boost-devel, zlib-devel, glibc-devel, python-devel
Requires:		python >= 2.4

%description
HOOMD-blue stands for Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition. It performs general purpose particle dynamics simulations on a single workstation, taking advantage of NVIDIA GPUs to attain a level of performance equivalent to many processor cores on a fast cluster.

Object-oriented design patterns are used in HOOMD-blue so it versatile and expandable. Various types of potentials, integration methods and file formats are currently supported, and more are added with each release. The code is available and open source, so anyone can write a plugin or change the source to add additional functionality.

Simulations are configured and run using simple python scripts, allowing complete control over the force field choice, integrator, all parameters, how many time steps are run, etc... . The scripting system is designed to be as simple as possible to the non-programmer.

%prep
rm -rf $RPM_BUILD_DIR/%{name}-%{version}

%setup -T -c
git clone %{repository} .
if [ $? -ne 0 ]; then
  exit $?
fi
git checkout %{refspec}
if [ $? -ne 0 ]; then
  exit $?
fi


cmake -DCMAKE_INSTALL_PREFIX=$RPM_BUILD_ROOT/usr -DLIB_SUFFIX=%{libsuffix} -DENABLE_EMBED_CUDA=ON -DPYTHON_SITEDIR=$RPM_BUILD_ROOT/%{sitedir} -DPYTHON_EXECUTABLE=%{python}

%build
cd $RPM_BUILD_DIR/%{name}-%{version}
make
make preinstall

%install
rm -rf $RPM_BUILD_ROOT

rm -f fluid-file-list
echo "%{sitedir}/hoomd.so" >> fluid-file-list
echo "%{sitedir}/hoomd_plugins" >> fluid-file-list
echo "%{sitedir}/hoomd_script" >> fluid-file-list
echo "/usr/%{libname}/hoomd" >> fluid-file-list

make install
%clean
rm -rf $RPM_BUILD_ROOT

%post

%postun

%files -f fluid-file-list
%defattr(-,root,root)
/usr/share/hoomd
/usr/bin/hoomd
/usr/bin/hoomd-config.sh
/usr/include/hoomd
/usr/share/icons/hicolor/48x48/mimetypes/application-x-hoomd.png
/usr/share/icons/hicolor/48x48/apps/hoomd.png
/usr/share/icons/hicolor/128x128/mimetypes/application-x-hoomd.png
/usr/share/icons/hicolor/128x128/apps/hoomd.png
/usr/share/icons/hicolor/32x32/mimetypes/application-x-hoomd.png
/usr/share/icons/hicolor/32x32/apps/hoomd.png
/usr/share/mime/packages/hoomd.xml
/usr/share/applications/HOOMD.desktop
