#update our codebase
cd  ../
fsvn up

#check  the previous version built
atrev=$(cat packaging/deb_old_version)
if [ "$atrev" -eq "$(svnversion . )" ];then
	echo "up to date"
else
#get what architecture we're building on.
	export deb_arch=$(dpkg-architecture -qDEB_BUILD_GNU_CPU)
	if [ "$(dpkg-architecture -qDEB_BUILD_ARCH_BITS )" == "64" ] 2>/dev/null ; 
		then export lib_suffix="64" ;	fi
#remove old traces of building packages
	rm -r debian/*
	echo $(svnversion .) > packaging/deb_old_version
#create build directory to move cuda libs to
	export deb_build_folder="obj-$(dpkg-architecture -qDEB_BUILD_GNU_CPU)-linux-gnu" 
	mkdir -p  debian/hoomd-blue/usr/lib${lib_suffix}

#make sure cuda libs are  where we want it
	cp -r packaging/debian ./
	mkdir -p debian/hoomd-blue/usr/lib${lib_suffix}
	cp /usr/local/cuda/lib${lib_suffix}/libcudart.so*  debian/hoomd-blue/usr/lib${lib_suffix}
	cp /usr/local/cuda/lib${lib_suffix}/libcudafft.so* debian/hoomd-blue/usr/lib${lib_suffix}


#export the variables to set the version from svn
	export HSVN_VERSION=$(svnversion . )
	export HVERSION="0.9.1."${HSVN_VERSION}
	echo $HVERSION
#set our package version in changelog
	sed s/HVERSION/${HVERSION}/ debian/changelog -i
	sed s/#M/#/ debian/changelog -i
#call package builder
	dpkg-buildpackage
fi
