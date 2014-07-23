BRANCH=maint

#update our codebase
cd  ../
git fetch
git checkout ${BRANCH}
git pull --ff-only

#check  the previous version built
export atrev="$(cat ../deb_old_version)"
export old_ver="$(git describe)"
echo $atrev
echo $old_ver
if [ "$atrev" =  "$old_ver" ];then
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
        sed -i s/amd64/i386/ debian/files

fi
#set the version we just setup for in deb_old_version so it won't be built again
    echo $(git describe) > ../deb_old_version

#export the variables to set the version from git
    export HVERSION_BASE=$(git describe | awk 'BEGIN { FS = "-" } ; {print $1}' | cut -c2-)

    export HREVISION=$(git describe | awk 'BEGIN { FS = "-" } ; {print $2}')
#set a zero revision if HREVISION is blank
    if [ -z "${HREVISION}" ]; then
        HREVISION="0"
    fi
    export HVERSION="${HVERSION_BASE}-"${HREVISION}
    echo $HVERSION
#set our package version in changelog
    sed s/HVERSION/${HVERSION}/ debian/changelog -i
#call package builder
    dpkg-buildpackage
#move files to be uploaded
    cd ..
    cp deb_old_version deb_version${lib_suffix}
    destination="daily/incoming/"`/usr/bin/lsb_release -d | /usr/bin/awk '{print $2$3$4}' FS="[\t .]" | tr '[:upper:]' '[:lower:]'`
    scp deb_version${lib_suffix} joaander@petry.engin.umich.edu:$destination
    scp hoomd-blue_${HVERSION}_$(dpkg-architecture -qDEB_BUILD_ARCH).deb joaander@petry.engin.umich.edu:$destination
fi
