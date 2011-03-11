#update our codebase
cd  ../
svn up

#check  the previous version built
export atrev="$(cat ../deb_old_version)"
export old_ver="$(svnversion . )"
echo $atrev
echo $old_ver
if [ $atrev =  $old_ver ];then
	echo "up to date"
else
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
		sed -i s/and64/i386/ debian/files

fi
#set the version we just setup for in deb_old_version so it won't be built again
	echo $(svnversion .) > ../deb_old_version

#export the variables to set the version from svn
	export HSVN_VERSION=$(svnversion . )
	export HVERSION="0.9.1."${HSVN_VERSION}
	echo $HVERSION
#set our package version in changelog
	sed s/HVERSION/${HVERSION}/ debian/changelog -i
#call package builder
	dpkg-buildpackage
#move files to be uploaded
	cd ..
	cp deb_old_version deb_version${lib_suffix}
	scp deb_version${lib_suffix} joaander@foxx.engin.umich.edu:devel/incoming
	scp hoomd-blue_${HVERSION}*.deb joaander@foxx.engin.umich.edu:devel/incoming
fi
