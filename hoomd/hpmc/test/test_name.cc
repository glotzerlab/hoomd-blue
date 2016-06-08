#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

#include <boost/bind.hpp>
#include <boost/python.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/function.hpp>
#include <memory>

//! Name the unit test module
#define BOOST_TEST_MODULE HelloWorld
#include "boost_utf_configure.h"


int main(int argc, char** argv)
    {
    std::cout<<"Hello"<<std::endl;
    }
