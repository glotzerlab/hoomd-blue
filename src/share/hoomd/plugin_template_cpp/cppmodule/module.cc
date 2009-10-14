// Include the defined classes that are to be exported to python
#include "ExampleUpdater.h"

// Include boost.python to do the exporting
#include <boost/python.hpp>
using namespace boost::python;

// specify the python module. Note that the name must expliclty match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
BOOST_PYTHON_MODULE(_plugin_template)
    {
    export_ExampleUpdater();
    
    #ifdef ENABLE_CUDA
    export_ExampleUpdaterGPU();
    #endif
    }
