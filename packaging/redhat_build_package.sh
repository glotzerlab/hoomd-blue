#!/bin/bash
# This script facilitates automatic nightly builds of RPM hoomd packages.
# To simply build a package, consider using 'make rpm' instead, or refer to
# the README or spec file for more information.
#
# To perform automatic package builds, this script should be invoked from cron
# with the following additions to crontab:
# MAILTO=''
# 0 0 * * * $HOME/hoomd-blue/packaging/redhat_build_package.sh >> $HOME/redhat_package_build.out 2>&1
#
# If this script is invoked automatically (as via crontab) from $HOME/hoomd-blue/packaging/
# then updates in hoomd-blue git will take two subsequent runs to take effect.
#
# Use the -f flag to prevent the script from performing an automatic update
# of itself and the Makefile.
#
BRANCH=maint
PATH=/bin:/usr/bin:$PATH
# $0 can't be relied upon to identify the name of this script...
ME=redhat_build_package.sh
SPECFILE="hoomd.spec"
# number of RPMs to retain
NRPMS=8
# what architecture does rpm think we have?
ARCH=`rpm --eval '%{_arch}'`

QUIET='QUIET=true'
UPDATE="true"
while getopts fv OPT $@; do
 case $OPT in
    'f') unset UPDATE ;;
    'v') unset QUIET ;;
 esac
done

echo "$0 running at "`date`

cd $HOME/hoomd-blue
git fetch
git checkout ${BRANCH}
git pull --ff-only
cd packaging

# determine version and revision
HVERSION_BASE=$(git describe | awk 'BEGIN { FS = "-" } ; {print $1}' | cut -c2-)

HREVISION=$(git describe | awk 'BEGIN { FS = "-" } ; {print $2}')
#set a zero revision if HREVISION is blank
if [ -z "${HREVISION}" ]; then
    HREVISION="0"
fi

#check the previous version built
atrev=`cat $HOME/rh_old_revsion || echo 0`
new_rev=`git describe`
echo "Last revision built was $atrev"
echo "Current repository revision is $new_rev"

if [ "$atrev" =  "$new_rev" ];then
    echo "up to date"
else
    echo "commence building"
    # maybe some of this should be moved to cmake
    mkdir -p $HOME/nightly-build
    cp Makefile $HOME/nightly-build/
    cp -R SPECS $HOME/nightly-build/
    cd $HOME/nightly-build

    make rpm VERSION=${HVERSION_BASE} RELEASE=${HREVISION} REFSPEC=${BRANCH} $QUIET || exit
    #set the version we just built in rh_old_revsion so it won't be built again
    echo $new_rev > $HOME/rh_old_revsion
    #move files to be uploaded
    if [ -e /etc/redhat-release ];then
        destination="devel/incoming/"`/bin/cat /etc/redhat-release | /usr/bin/awk '{print $1$3}' FS="[ .]" | tr '[:upper:]' '[:lower:]'`
    else
        # assume that this is opensuse and format accordingly
        destination="devel/incoming/"`/usr/bin/lsb_release -d | /usr/bin/awk '{print $2$3$4}' FS="[\t .]" | tr '[:upper:]' '[:lower:]'`
    fi
    "rsync -ue /usr/bin/ssh rpmbuild/RPMS/$ARCH/hoomd*.rpm joaander@foxx.engin.umich.edu:$destination/"
fi

#clean up
cd $HOME/nightly-build/rpmbuild/RPMS/$ARCH
rpmfiles=( `ls -td hoomd*.rpm` )
numfiles=${#rpmfiles[*]}
for ((  i=$(( $NRPMS )); $i < $numfiles ; i++ )); do
    rm ${rpmfiles[$i]};
done
