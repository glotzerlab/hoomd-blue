echo "--- Building nightly Mac OS X hoomd package on `date`"

# get up to the root of the tree
cd ..
cd ..

# update to the latest rev
svn update

# up another level and redo the build directory
cd ..
rm -rf build
mkdir build
cd build

cmake -DENABLE_DOXYGEN=OFF -DENABLE_APP_BUNDLE_INSTALL=ON -DBOOST_ROOT=/opt/boost-1.46.0/ -DBoost_NO_SYSTEM_PATHS=ON -DPYTHON_EXECUTABLE=/usr/bin/python ../hoomd/

make package -j6
mv hoomd-blue-*.dmg /Users/joaander/devel/incoming/mac
echo "--- Done!"
echo ""
