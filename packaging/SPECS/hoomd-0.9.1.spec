%global release		%(svn info http://codeblue.engin.umich.edu/hoomd-blue/svn/trunk|grep Revision |awk '{print $2}')
%global libname		lib%(uname -m |sed -n 's/.*64$/64/p')
%define pyver		%( rpm -q --qf \%\{version\} python |awk -F. '{print $1"."$2}' )
%global python			python%{pyver}

BuildRoot:		%{_tmppath}/%{name}-root
Summary: 		Highly Object Oriented Molecular Dynamics package
License: 		various
Name: 			hoomd
Version: 		0.9.1
Release: 		%{release}
# sources will be retrieved with subversion
# Source: 		http://codeblue.umich.edu/hoomd-blue/downloads/0.9/hoomd-0.9.1.tar.bz2
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
svn checkout http://codeblue.engin.umich.edu/hoomd-blue/svn/trunk .
if [ $? -ne 0 ]; then
  exit $?
fi

cmake -DCMAKE_INSTALL_PREFIX=$RPM_BUILD_ROOT/usr -DENABLE_NATIVE_INSTALL=ON -DPYTHON_SITEDIR=%{libname}/%{python}/site-packages -DPYTHON_INCLUDE_DIR=/usr/include/%{python} -DPYTHON_LIBRARY=/usr/%{libname}/lib%{python}.so.1.0

%build
cd $RPM_BUILD_DIR/%{name}-%{version}
make
make preinstall

%install
CUDART=`ldd libhoomd/hoomd.so |grep cudart|awk '{print $3}'`
LOCALCUDA=%{prefix}/%{libname}/`basename $CUDART`
HOOMDCUDA=%{prefix}/%{libname}/hoomd/`basename $CUDART`
rm -rf $RPM_BUILD_ROOT

rm -f fluid-file-list
echo "/usr/%{libname}/%{python}/site-packages/hoomd.so" >> fluid-file-list
echo "/usr/%{libname}/%{python}/site-packages/hoomd_plugins" >> fluid-file-list
echo "/usr/%{libname}/%{python}/site-packages/hoomd_script" >> fluid-file-list

mkdir -p $RPM_BUILD_ROOT/usr/%{libname}/hoomd/
cp $CUDART $RPM_BUILD_ROOT/usr/%{libname}/hoomd/
cat > $RPM_BUILD_ROOT/usr/%{libname}/hoomd/mklink <<EOF
if [ ! -e $LOCALCUDA ] ; then
	ln -s $HOOMDCUDA $LOCALCUDA
fi
EOF
cat > $RPM_BUILD_ROOT/usr/%{libname}/hoomd/rmlink <<EOF
if [ -L $LOCALCUDA ]
	if ( ls -l $LOCALCUDA |grep -q $HOOMDCUDA ) ;
		then rm $LOCALCUDA
	fi
fi
EOF
echo "/usr/%{libname}/hoomd" >> fluid-file-list

make install
%clean
rm -rf $RPM_BUILD_ROOT

%post
sh %{prefix}/%{libname}/hoomd/mklink

%postun
sh %{prefix}/%{libname}/hoomd/rmlink

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
