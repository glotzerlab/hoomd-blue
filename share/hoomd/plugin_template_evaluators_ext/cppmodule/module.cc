// Include the defined classes that are to be exported to python
#include "AllPairExtPotentials.h"
#include "AllBondExtPotentials.h"

// Include boost.python to do the exporting
#include <boost/python.hpp>
using namespace boost::python;

// specify the python module. Note that the name must expliclty match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
BOOST_PYTHON_MODULE(_evaluators_ext_template)
    {
    // export pair potential
    export_PotentialPair<PotentialPairLJ2>("PotentialPairLJ2");

    // export bond potential
    export_PotentialBond<PotentialBondHarmonicDPD>("PotentialBondHarmonicDPD");
    #ifdef ENABLE_CUDA
    export_PotentialPairGPU<PotentialPairLJ2GPU, PotentialPairLJ2>("PotentialPairLJ2GPU");
    export_PotentialBondGPU<PotentialBondHarmonicDPDGPU, PotentialBondHarmonicDPD>("PotentialBondHarmonicDPDGPU");
    #endif
    }
