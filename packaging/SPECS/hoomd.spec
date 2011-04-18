%global	repository	http://codeblue.engin.umich.edu/hoomd-blue/svn/
# A tagged hoomd release version number may be specified by defining %{version}.
# A subversion revision number may be specified by defining %{release}.
# Otherwise %{release} is the last revision affecting trunk.
# Note that if both version and release are specified, the latter is ignored.
%if %{?version:1}%{!?version:0}
%global branch	tags/hoomd-%{version}
%global release	%(svn info %{repository}%{branch} |grep 'Last Changed Rev' |awk '{print $NF}')
%else
%global	branch	trunk
%endif
%global version	%{?version}%{!?version:0.9.2}
%global release	%{?release}%{!?release:%(svn info %{repository}%{branch} |grep 'Last Changed Rev' |awk '{print $NF}')}

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
# Source: 		http://codeblue.umich.edu/hoomd-blue/downloads/0.9/hoomd-0.9.2.tar.bz2
URL:			http://codeblue.umich.edu/hoomd-blue/
Prefix:			/usr
Group: 			Applications
BuildRequires:		gcc-c++, boost-devel, zlib-devel, glibc-devel, python-devel
Requires:		python >= 2.4, boost >= 1.32

%description
HOOMD-blue stands for Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition. It performs general purpose particle dynamics simulations on a single workstation, taking advantage of NVIDIA GPUs to attain a level of performance equivalent to many processor cores on a fast cluster.

Object-oriented design patterns are used in HOOMD-blue so it versatile and expandable. Various types of potentials, integration methods and file formats are currently supported, and more are added with each release. The code is available and open source, so anyone can write a plugin or change the source to add additional functionality.

Simulations are configured and run using simple python scripts, allowing complete control over the force field choice, integrator, all parameters, how many time steps are run, etc... . The scripting system is designed to be as simple as possible to the non-programmer.

HOOMD-blue is a direct continuation of the project HOOMD, originally developed at Ames Lab and Iowa State University: http://www.ameslab.gov/hoomd. The HOOMD-blue lead development effort is at the University of Michigan, though the software has many contributors from around the world. The "blue" suffix is actually part of the acronym and stands for Blue Edition, a subtle hint to one of the University of Michigan school colors.

%prep
rm -rf $RPM_BUILD_DIR/%{name}-%{version}

%setup -T -c
svn checkout -r %{release} %{repository}%{branch} .
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
