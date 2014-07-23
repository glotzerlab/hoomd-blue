# number of builds to retain
NRPMS=8
BRANCH=master

echo "--- Building nightly Mac OS X hoomd package on `date`"

# get up to the root of the tree
cd ..
cd ..

# update to the latest rev
git fetch
git checkout ${BRANCH}
git pull --ff-only

new_rev=`git describe`

# up another level out of the source repository
cd ..

#check the previous version built
atrev=`cat mac_old_revision || echo 0`
echo "Last revision built was $atrev"
echo "Current repository revision is $new_rev"

if [ "$atrev" =  "$new_rev" ];then
    echo "up to date"
else
    echo $new_rev > mac_old_revision

    # build the new package
    rm -rf build
    mkdir build
    cd build

    cmake -DENABLE_MPI=OFF -DENABLE_DOXYGEN=OFF -DENABLE_APP_BUNDLE_INSTALL=ON -DBOOST_ROOT=/opt/boost-1.52.0/ -DBoost_NO_SYSTEM_PATHS=ON -DPYTHON_EXECUTABLE=/usr/bin/python ../code

    make package -j6
    destination="daily/incoming/mac"
    rsync -ue /usr/bin/ssh hoomd-*.dmg joaander@petry.engin.umich.edu:$destination/
fi

rpmfiles=( `ls -td hoomd-*.dmg` )
numfiles=${#rpmfiles[*]}
for ((  i=$(( $NRPMS )); $i < $numfiles ; i++ )); do
    rm ${rpmfiles[$i]};
done

echo "--- Done!"
echo ""
