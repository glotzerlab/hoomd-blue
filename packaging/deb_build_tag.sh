#move up two directories and check out tag.
cd  ../..
svn co http://codeblue.umich.edu/hoomd-blue/svn/tags/hoomd-$1
cd hoomd-$1

echo "commence building"
rm -r debian/*
cp -r packaging/debian ./
#get what architecture we're building on.
if [ `arch` = "x86_64" ]
	then
#patch debian folder for 64 bit
	echo "patching 64bit"
	export lib_suffix="64"
	sed -i s/i386/amd64/ debian/control
	sed -i s/i386/amd64/ debian/files
else
#patch debian folder for building on i386
	echo "patching 32bit"
	export lib_suffix=""
	sed -i s/amd64/i386/ debian/control
	sed -i s/amd64/i386/ debian/files

fi
export HVERSION=$1
#set our package version in changelog
sed s/HVERSION/${HVERSION}/ debian/changelog -i
#call package builder
dpkg-buildpackage
#move files to be uploaded
cd ..
cp deb_old_version deb_version${lib_suffix}
#	scp deb_version${lib_suffix} joaander@foxx.engin.umich.edu:devel/incoming/ubuntu
#	scp hoomd-blue_${HVERSION}_$(dpkg-architecture -qDEB_BUILD_ARCH).deb joaander@foxx.engin.umich.edu:devel/incoming/ubuntu
