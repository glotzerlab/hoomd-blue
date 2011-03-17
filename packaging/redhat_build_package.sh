#!/bin/bash
# To perform automatic package builds, this script could be invoked from cron
# with the following additions to crontab:
# MAILTO=
# 0 0 * * * $HOME/packaging/redhat_build_package.sh >> $HOME/redhat_package_build.out 2>&1
#
# If this script is invoked automatically (as via crontab) from $HOME/packaging/
# then updates in hoomd-blue trunk will take two subsequent runs to take effect.
#
PATH=/bin:/usr/bin:$PATH
SPECFILE="hoomd-0.9.1.spec"

echo "$0 running at "`date`

#update our codebase
mkdir -p $HOME/packaging
cd $HOME/packaging
old_rev=`svnversion .`
if [ "exported" = "$old_rev" ] ; then
	svn checkout http://codeblue.engin.umich.edu/hoomd-blue/svn/trunk/packaging .
else
	svn update
fi

#check the previous version built
atrev=`cat $HOME/rh_old_revsion || echo 0`
new_rev=`svnversion .`
echo "Last revision built was $atrev"
echo "Current repository revision is $new_rev"

if [ "$atrev" =  "$new_rev" ];then
	echo "up to date"
else
	echo "commence building"
	#prepare build environment
	mkdir -p $HOME/rpmbuild/{SPECS,SOURCES,RPMS,SRPMS,BUILD}
	cp $HOME/packaging/SPECS/$SPECFILE $HOME/rpmbuild/SPECS/
	#call package builder
	PATH=$PATH:/usr/local/cuda/bin rpmbuild -ba --define "_topdir $HOME/rpmbuild" --define 'debug_package %{nil}' $HOME/rpmbuild/SPECS/$SPECFILE
	#set the version we just built in rh_old_revsion so it won't be built again
	echo $new_rev > $HOME/rh_old_revsion
	#move files to be uploaded
	destination = `/bin/cat /etc/redhat-release | /usr/bin/awk '{print $1}'``/bin/uname -p`
	echo rsync -ue /usr/bin/ssh $HOME/rpmbuild/RPMS/hoomd*.rpm joaander@foxx.engin.umich.edu:$destination
fi
