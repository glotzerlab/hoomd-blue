#ifndef NUM_UTIL_H__
#define NUM_UTIL_H__

// Copyright 2006  Phil Austin (http://www.eos.ubc.ca/personal/paustin)
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

/*! \file num_util.h
    \brief Helper routines for numpy arrays
*/

//
// $Id: num_util.h 39 2007-02-01 02:54:54Z phil $
//

#include <boost/python.hpp>
#include <numpy/noprefix.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <numeric>
#include <map>
#include <complex>



namespace num_util{
  //!
  /**
   *A free function that extracts a PyArrayObject from any sequential PyObject.
   *@param x a sequential PyObject wrapped in a Boost/Python 'object'.
   *@return a PyArrayObject wrapped in Boost/Python numeric array.
   */
  boost::python::numeric::array makeNum(boost::python::object x);

  /**
   *Creates an one-dimensional numpy array of length n and numpy type t.
   * The elements of the array are initialized to zero.
   *@param n an integer representing the length of the array.
   *@param t elements' numpy type. Default is double.
   *@return a numeric array of size n with elements initialized to zero.
   */
  boost::python::numeric::array makeNum(intp n, NPY_TYPES t);

  /**
   *Creates a n-dimensional numpy array with dimensions dimens and numpy
   *type t. The elements of the array are initialized to zero.
   *@param dimens a vector of interger specifies the dimensions of the array.
   *@param t elements' numpy type. Default is double.
   *@return a numeric array of shape dimens with elements initialized to zero.
   */
  boost::python::numeric::array makeNum(std::vector<intp> dimens,
                    NPY_TYPES t);

  /**
   *Function template returns NPY_Type for C++ type
   *See num_util.cpp for specializations
   *@param T C++ type
   *@return numpy type enum
   */

  template<typename T> NPY_TYPES getEnum(void)
  {
    PyErr_SetString(PyExc_ValueError, "no mapping available for this type");
    boost::python::throw_error_already_set();
    return NPY_VOID;
  }

    //specializations for use by makeNum


  template <>
  NPY_TYPES getEnum<unsigned char>(void);

  template <>
  NPY_TYPES getEnum<signed char>(void);

  template <>
  NPY_TYPES getEnum<short>(void);

  template <>
  NPY_TYPES getEnum<unsigned short>(void);

  template <>
  NPY_TYPES getEnum<unsigned int>(void);

  template <>
  NPY_TYPES getEnum<int>(void);

  template <>
  NPY_TYPES getEnum<long>(void);

  template <>
  NPY_TYPES getEnum<unsigned long>(void);

  template <>
  NPY_TYPES getEnum<long long>(void);

  template <>
  NPY_TYPES getEnum<unsigned long long>(void);

  template <>
  NPY_TYPES getEnum<float>(void);

  template <>
  NPY_TYPES getEnum<double>(void);

  template <>
  NPY_TYPES getEnum<long double>(void);

  template <>
  NPY_TYPES getEnum<std::complex<float> >(void);

  template <>
  NPY_TYPES getEnum<std::complex<double> >(void);

  template <>
  NPY_TYPES getEnum<std::complex<long double> >(void);


  /**
   *Function template creates a one-dimensional numpy array of length n containing
   *a copy of data at data*.  See num_util.cpp::getEnum<T>() for list of specializations
   *@param T  C type of data
   *@param T* data pointer to start of data
   *@param n an integer indicates the size of the array.
   *@return a numpy array of size n with elements initialized to data.
   */

  template <typename T> boost::python::numeric::array makeNum(T* data, intp n = 0){
    boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(1, &n, getEnum<T>())));
    void *arr_data = PyArray_DATA((PyArrayObject*) obj.ptr());
    memcpy(arr_data, data, PyArray_ITEMSIZE((PyArrayObject*) obj.ptr()) * n); // copies the input data to
    return boost::python::extract<boost::python::numeric::array>(obj);
  }

  /**
   *Function template creates an n-dimensional numpy array with dimensions dimens containing
   *a copy of values starting at data.  See num_util.cpp::getEnum<T>() for list of specializations
   *@param T  C type of data
   *@param T*  data pointer to start of data
   *@param n an integer indicates the size of the array.
   *@return a numpy array of size n with elements initialized to data.
   */


  template <typename T> boost::python::numeric::array makeNum(T * data, std::vector<intp> dims){
    intp total = std::accumulate(dims.begin(),dims.end(),1,std::multiplies<intp>());
    boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(dims.size(),&dims[0], getEnum<T>())));
    void *arr_data = PyArray_DATA((PyArrayObject*) obj.ptr());
    memcpy(arr_data, data, PyArray_ITEMSIZE((PyArrayObject*) obj.ptr()) * total);
    return boost::python::extract<boost::python::numeric::array>(obj);
  }


  /**
   *Creates a numpy array from a numpy array, referencing the data.
   *@param arr a Boost/Python numeric array.
   *@return a numeric array referencing the input array.
   */
  boost::python::numeric::array makeNum(const
                    boost::python::numeric::array& arr);

  /**
   *A free function that retrieves the numpy type of a numpy array.
   *@param arr a Boost/Python numeric array.
   *@return the numpy type of the array's elements
   */
  NPY_TYPES type(boost::python::numeric::array arr);

  /**
   *Throws an exception if the actual array type is not equal to the expected
   *type.
   *@param arr a Boost/Python numeric array.
   *@param expected_type an expected numpy type.
   *@return -----
   */
  void check_type(boost::python::numeric::array arr,
          NPY_TYPES expected_type);

  /**
   *A free function that retrieves the number of dimensions of a numpy array.
   *@param arr a Boost/Python numeric array.
   *@return an integer that indicates the rank of an array.
   */
  int rank(boost::python::numeric::array arr);

  /**
   *Throws an exception if the actual rank is not equal to the expected rank.
   *@param arr a Boost/Python numeric array.
   *@param expected_rank an expected rank of the numeric array.
   *@return -----
   */
  void check_rank(boost::python::numeric::array arr, int expected_rank);

  /**
   *A free function that returns the total size of the array.
   *@param arr a Boost/Python numeric array.
   *@return an integer that indicates the total size of the array.
   */
  intp size(boost::python::numeric::array arr);

  /**
   *Throw an exception if the actual total size of the array is not equal to
   *the expected size.
   *@param arr a Boost/Python numeric array.
   *@param expected_size the expected size of an array.
   *@return -----
   */
  void check_size(boost::python::numeric::array arr, intp expected_size);

  /**
   *Returns the dimensions in a vector.
   *@param arr a Boost/Python numeric array.
   *@return a vector with integer values that indicates the shape of the array.
  */
  std::vector<intp> shape(boost::python::numeric::array arr);

  /**
   *Returns the size of a specific dimension.
   *@param arr a Boost/Python numeric array.
   *@param dimnum an integer that identifies the dimension to retrieve.
   *@return the size of the requested dimension.
   */
  intp get_dim(boost::python::numeric::array arr, int dimnum);

  /**
   *Throws an exception if the actual dimensions of the array are not equal to
   *the expected dimensions.
   *@param arr a Boost/Python numeric array.
   *@param expected_dims an integer vector of expected dimension.
   *@return -----
   */
  void check_shape(boost::python::numeric::array arr,
           std::vector<intp> expected_dims);

  /**
   *Throws an exception if a specific dimension from a numpy array does not
   *match the expected size.
   *@param arr a Boost/Python numeric array.
   *@param dimnum an integer that specifies which dimension of 'arr' to check.
   *@param dimsize an expected size of the specified dimension.
   *@return -----
  */
  void check_dim(boost::python::numeric::array arr, int dimnum, intp dimsize);

  /**
   *Returns true if the array is contiguous.
   *@param arr a Boost/Python numeric array.
   *@return true if the array is contiguous, false otherwise.
  */
  bool iscontiguous(boost::python::numeric::array arr);

  /**
   *Throws an exception if the array is not contiguous.
   *@param arr a Boost/Python numeric array.
   *@return -----
  */
  void check_contiguous(boost::python::numeric::array arr);

  /**
   *Returns a pointer to the data in the array.
   *@param arr a Boost/Python numeric array.
   *@return a char pointer pointing at the first element of the array.
   */
  void* data(boost::python::numeric::array arr);

  /**
   *Copies data into the array.
   *@param arr a Boost/Python numeric array.
   *@param new_data a char pointer referencing the new data.
   *@return -----
   */
  void copy_data(boost::python::numeric::array arr, char* new_data);

  /**
   *Returns a clone of this array.
   *@param arr a Boost/Python numeric array.
   *@return a replicate of the Boost/Python numeric array.
   */
  boost::python::numeric::array clone(boost::python::numeric::array arr);

  /**
   *Returns a clone of this array with a new type.
   *@param arr a Boost/Python numeric array.
   *@param t NPY_TYPES of the output array.
   *@return a replicate of 'arr' with type set to 't'.
   */
  boost::python::numeric::array astype(boost::python::numeric::array arr,
                       NPY_TYPES t);


/*    *Returns the reference count of the array. */
/*    *@param arr a Boost/Python numeric array. */
/*    *@return the reference count of the array. */

  int refcount(boost::python::numeric::array arr);

  /**
   *Returns the strides array in a vector of integer.
   *@param arr a Boost/Python numeric array.
   *@return the strides of an array.
   */
  std::vector<intp> strides(boost::python::numeric::array arr);

  /**
   *Throws an exception if the element of a numpy array is type cast to
   *NPY_OBJECT.
   *@param newo a Boost/Python object.
   *@return -----
   */
  void check_PyArrayElementType(boost::python::object newo);

  /**
   *Mapping from a NPY_TYPE to its corresponding name in string.
   */
  typedef std::map<NPY_TYPES, std::string> KindStringMap;

  /**
   *Mapping from a NPY_TYPE to its corresponding typeID in char.
   */
  typedef std::map<NPY_TYPES, char> KindCharMap;

  /**
   *Mapping from a typeID to its corresponding NPY_TYPE.
   */
  typedef std::map<char, NPY_TYPES> KindTypeMap;

  /**
   *Converts a NPY_TYPE to its name in string.
   *@param t_type a NPY_TYPES.
   *@return the corresponding name in string.
   */
  std::string type2string(NPY_TYPES t_type);

  /**
   *Converts a NPY_TYPE to its single character typecode.
   *@param t_type a NPY_TYPES.
   *@return the corresponding typecode in char.
   */
  char type2char(NPY_TYPES t_type);

  /**
   *Coverts a single character typecode to its NPY_TYPES.
   *@param e_type a NPY_TYPES typecode in char.
   *@return its corresponding NPY_TYPES.
   */
  NPY_TYPES char2type(char e_type);

  /**
   *Constructs a string which contains a list of elements extracted from the
   *input vector.
   *@param vec a vector of any type.
   *@return a string that lists the elements from the input vector.
   */
  template <class T>
  inline std::string vector_str(const std::vector<T>& vec);

  /**
   *Throws an exception if the total size computed from a vector of integer
   *does not match with the expected size.
   *@param dims an integer vector of dimensions.
   *@param n an expected size.
   *@return -----
   */
  inline void check_size_match(std::vector<intp> dims, intp n);

} //  namespace num_util

#endif
