#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

#include <boost/bind.hpp>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <boost/test/unit_test.hpp>
#include <boost/function.hpp>
#include <memory>

//! Name the unit test module
UP_TEST(HelloWorld)
#include "hoomd/test/upp11_config.h"


int main(int argc, char** argv)
    {
    std::cout<<"Hello"<<std::endl;
    }


