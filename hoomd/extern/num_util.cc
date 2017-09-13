// Copyright 2006  Phil Austin (http://www.eos.ubc.ca/personal/paustin)
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

/*! \file num_util.cc
    \brief Helper routines for numpy arrays
*/

#include "num_util.h"

namespace num_util{

  //specializations for use by makeNum


  template <>
  NPY_TYPES getEnum<unsigned char>(void)
  {
    return NPY_UBYTE;
  }


  template <>
  NPY_TYPES getEnum<signed char>(void)
  {
    return NPY_BYTE;
  }

  template <>
  NPY_TYPES getEnum<short>(void)
  {
    return NPY_SHORT;
  }

  template <>
  NPY_TYPES getEnum<unsigned short>(void)
  {
    return NPY_USHORT;
  }


  template <>
  NPY_TYPES getEnum<unsigned int>(void)
  {
    return NPY_UINT;
  }

  template <>
  NPY_TYPES getEnum<int>(void)
  {
    return NPY_INT;
  }

  template <>
  NPY_TYPES getEnum<long>(void)
  {
    return NPY_LONG;
  }

  template <>
  NPY_TYPES getEnum<unsigned long>(void)
  {
    return NPY_ULONG;
  }


  template <>
  NPY_TYPES getEnum<long long>(void)
  {
    return NPY_LONGLONG;
  }

  template <>
  NPY_TYPES getEnum<unsigned long long>(void)
  {
    return NPY_ULONGLONG;
  }

  template <>
  NPY_TYPES getEnum<float>(void)
  {
    return NPY_FLOAT;
  }

  template <>
  NPY_TYPES getEnum<double>(void)
  {
    return NPY_DOUBLE;
  }

  template <>
  NPY_TYPES getEnum<long double>(void)
  {
    return NPY_LONGDOUBLE;
  }

  template <>
  NPY_TYPES getEnum<std::complex<float> >(void)
  {
    return NPY_CFLOAT;
  }


  template <>
  NPY_TYPES getEnum<std::complex<double> >(void)
  {
    return NPY_CDOUBLE;
  }

  template <>
  NPY_TYPES getEnum<std::complex<long double> >(void)
  {
    return NPY_CLONGDOUBLE;
  }


typedef KindStringMap::value_type  KindStringMapEntry;
KindStringMapEntry kindStringMapEntries[] =
  {
    KindStringMapEntry(NPY_UBYTE,  "NPY_UBYTE"),
    KindStringMapEntry(NPY_BYTE,   "NPY_BYTE"),
    KindStringMapEntry(NPY_SHORT,  "NPY_SHORT"),
    KindStringMapEntry(NPY_INT,    "NPY_INT"),
    KindStringMapEntry(NPY_LONG,   "NPY_LONG"),
    KindStringMapEntry(NPY_FLOAT,  "NPY_FLOAT"),
    KindStringMapEntry(NPY_DOUBLE, "NPY_DOUBLE"),
    KindStringMapEntry(NPY_CFLOAT, "NPY_CFLOAT"),
    KindStringMapEntry(NPY_CDOUBLE,"NPY_CDOUBLE"),
    KindStringMapEntry(NPY_OBJECT, "NPY_OBJECT"),
    KindStringMapEntry(NPY_NTYPES, "NPY_NTYPES"),
    KindStringMapEntry(NPY_NOTYPE ,"NPY_NOTYPE")
  };

typedef KindCharMap::value_type  KindCharMapEntry;
KindCharMapEntry kindCharMapEntries[] =
  {
    KindCharMapEntry(NPY_UBYTE,  'B'),
    KindCharMapEntry(NPY_BYTE,   'b'),
    KindCharMapEntry(NPY_SHORT,  'h'),
    KindCharMapEntry(NPY_INT,    'i'),
    KindCharMapEntry(NPY_LONG,   'l'),
    KindCharMapEntry(NPY_FLOAT,  'f'),
    KindCharMapEntry(NPY_DOUBLE, 'd'),
    KindCharMapEntry(NPY_CFLOAT, 'F'),
    KindCharMapEntry(NPY_CDOUBLE,'D'),
    KindCharMapEntry(NPY_OBJECT, 'O')
  };

typedef KindTypeMap::value_type  KindTypeMapEntry;
KindTypeMapEntry kindTypeMapEntries[] =
  {
    KindTypeMapEntry('B',NPY_UBYTE),
    KindTypeMapEntry('b',NPY_BYTE),
    KindTypeMapEntry('h',NPY_SHORT),
    KindTypeMapEntry('i',NPY_INT),
    KindTypeMapEntry('l',NPY_LONG),
    KindTypeMapEntry('f',NPY_FLOAT),
    KindTypeMapEntry('d',NPY_DOUBLE),
    KindTypeMapEntry('F',NPY_CFLOAT),
    KindTypeMapEntry('D',NPY_CDOUBLE),
    KindTypeMapEntry('O',NPY_OBJECT)
  };


int numStringEntries = sizeof(kindStringMapEntries)/sizeof(KindStringMapEntry);
int numCharEntries = sizeof(kindCharMapEntries)/sizeof(KindCharMapEntry);
int numTypeEntries = sizeof(kindTypeMapEntries)/sizeof(KindTypeMapEntry);


static KindStringMap kindstrings(kindStringMapEntries,
                                   kindStringMapEntries + numStringEntries);

static KindCharMap kindchars(kindCharMapEntries,
                                   kindCharMapEntries + numCharEntries);

static KindTypeMap kindtypes(kindTypeMapEntries,
                                   kindTypeMapEntries + numTypeEntries);

//Create a numarray referencing Python sequence object
PyObject* makeNum(PyObject *x){
  if (!PySequence_Check(x)){
    PyErr_SetString(PyExc_ValueError, "expected a sequence");
    throw pybind11::error_already_set();
  }
  PyObject *obj = PyArray_ContiguousFromObject(x,NPY_NOTYPE,0,0);
  check_PyArrayElementType(obj);
  return obj;
}

//Create a one-dimensional PyArrayObject array of length n and Numeric type t
PyObject* makeNum(intp n, NPY_TYPES t=NPY_DOUBLE){
  PyObject* obj = PyArray_SimpleNew(1, &n, t);
  void *arr_data = PyArray_DATA((PyArrayObject*) obj);
  memset(arr_data, 0, PyArray_ITEMSIZE((PyArrayObject*) obj) * n);
  return obj;
}

//Create a PyArrayObject array with dimensions dimens and Numeric type t
PyObject* makeNum(std::vector<intp> dimens,
               NPY_TYPES t=NPY_DOUBLE){
  intp total = std::accumulate(dimens.begin(),dimens.end(),1,std::multiplies<intp>());
  PyObject* obj = PyArray_SimpleNew(dimens.size(), &dimens[0], t);
  void *arr_data = PyArray_DATA((PyArrayObject*) obj);
  memset(arr_data, 0, PyArray_ITEMSIZE((PyArrayObject*) obj) * total);
  return obj;
}

NPY_TYPES type(PyObject *arr){
  return NPY_TYPES(PyArray_TYPE((PyArrayObject*)arr));
}

void check_is_PyArray(PyObject *arr) {
  if(!PyArray_Check(arr)){
    PyErr_SetString(PyExc_ValueError, "expected a PyArrayObject");
    throw pybind11::error_already_set();
  }
}


void check_type(PyObject *arr,
        NPY_TYPES expected_type){
  NPY_TYPES actual_type = type(arr);
  if (actual_type != expected_type) {
    std::ostringstream stream;
    stream << "expected Numeric type " << kindstrings[expected_type]
       << ", found Numeric type " << kindstrings[actual_type] << std::ends;
    PyErr_SetString(PyExc_TypeError, stream.str().c_str());
    throw pybind11::error_already_set();
  }
  return;
}

//Return the number of dimensions
int rank(PyObject *arr){
  //std::cout << "inside rank" << std::endl;
  check_is_PyArray(arr);
  return PyArray_NDIM((PyArrayObject*)arr);
}

//Throws an error if expected_rank does not equal
//the number of dimensions in arr.
void check_rank(PyObject *arr, int expected_rank){
  int actual_rank = rank(arr);
  if (actual_rank != expected_rank) {
    std::ostringstream stream;
    stream << "expected rank " << expected_rank
       << ", found rank " << actual_rank << std::ends;
    PyErr_SetString(PyExc_RuntimeError, stream.str().c_str());
    throw pybind11::error_already_set();
  }
  return;
}

//Returns total number of elements in arr
intp size(PyObject *arr)
{
  check_is_PyArray(arr);
  return PyArray_Size(arr);
}

//Throws an error if expected_size does not equal
//the number of elements in arr
void check_size(PyObject *arr, intp expected_size){
  intp actual_size = size(arr);
  if (actual_size != expected_size) {
    std::ostringstream stream;
    stream << "expected size " << expected_size
       << ", found size " << actual_size << std::ends;
    PyErr_SetString(PyExc_RuntimeError, stream.str().c_str());
    throw pybind11::error_already_set();
  }
  return;
}

//Returns the dimensions of arr in a std::vector object
std::vector<intp> shape(PyObject *arr){
  std::vector<intp> out_dims;
  check_is_PyArray(arr);
  intp* dims_ptr = PyArray_DIMS((PyArrayObject*)arr);
  int the_rank = rank(arr);
  for (int i = 0; i < the_rank; i++){
    out_dims.push_back(*(dims_ptr + i));
  }
  return out_dims;
}

//Returns the size of the given dimension in arr
intp get_dim(PyObject *arr, int dimnum){
  int the_rank=rank(arr);
  if(dimnum >= 0 && the_rank < dimnum){
    std::ostringstream stream;
    stream << "Error: asked for length of dimension ";
    stream << dimnum << " but rank of array is " << the_rank << std::ends;
    PyErr_SetString(PyExc_RuntimeError, stream.str().c_str());
    throw pybind11::error_already_set();
  }
  std::vector<intp> actual_dims = shape(arr);
  return actual_dims[dimnum];
}

//Throws an error if the given dimensions do not match the
//shape of the given array object
void check_shape(PyObject *arr, std::vector<intp> expected_dims){
  std::vector<intp> actual_dims = shape(arr);
  if (actual_dims != expected_dims) {
    std::ostringstream stream;
    stream << "expected dimensions " << vector_str(expected_dims)
       << ", found dimensions " << vector_str(actual_dims) << std::ends;
    PyErr_SetString(PyExc_RuntimeError, stream.str().c_str());
    throw pybind11::error_already_set();
  }
  return;
}

//Throws an error if the size of the given dimension in the given array
//does not equal the given dimension size
void check_dim(PyObject *arr, int dimnum, intp dimsize){
  std::vector<intp> actual_dims = shape(arr);
  if(actual_dims[dimnum] != dimsize){
    std::ostringstream stream;
    stream << "Error: expected dimension number ";
    stream << dimnum << " to be length " << dimsize;
    stream << ", but found length " << actual_dims[dimnum]  << std::ends;
    PyErr_SetString(PyExc_RuntimeError, stream.str().c_str());
    throw pybind11::error_already_set();
  }
  return;
}

//Returns true if the given array is C-style contiguous.
//Otherwise, returns false.
bool iscontiguous(PyObject *arr)
{
  check_is_PyArray(arr);
  return PyArray_ISCONTIGUOUS((PyArrayObject*)arr);
}

//Throws an error if the given array is not C-style contiguous.
void check_contiguous(PyObject *arr)
{
  if (!iscontiguous(arr)) {
    PyErr_SetString(PyExc_RuntimeError, "expected a contiguous array");
    throw pybind11::error_already_set();
  }
  return;
}

//Returns the data in the given array
void* data(PyObject *arr){
  check_is_PyArray(arr);
  return PyArray_DATA((PyArrayObject*)arr);
}

//Copy data into the array
void copy_data(PyObject *arr, char* new_data){
  char* arr_data = (char*) data(arr);
  intp nbytes = PyArray_NBYTES((PyArrayObject*)arr);
  for (intp i = 0; i < nbytes; i++) {
    arr_data[i] = new_data[i];
  }
  return;
}

//Return a clone of this array
PyObject* clone(PyObject *arr){
  return PyArray_NewCopy((PyArrayObject*)arr,NPY_CORDER);
}


//Returns the stride of the given array
std::vector<intp> strides(PyObject *arr){
  std::vector<intp> out_strides;
  check_is_PyArray(arr);
  intp* strides_ptr = PyArray_STRIDES((PyArrayObject*)arr);
  intp the_rank = rank(arr);
  for (intp i = 0; i < the_rank; i++){
    out_strides.push_back(*(strides_ptr + i));
  }
  return out_strides;
}

int refcount(PyObject *arr){
  return REFCOUNT(arr);
}

void check_PyArrayElementType(PyObject *obj){
  check_is_PyArray(obj);
  NPY_TYPES theType=NPY_TYPES(PyArray_TYPE((PyArrayObject*)obj));
  if(theType == NPY_OBJECT){
      std::ostringstream stream;
      stream << "array elments have been cast to NPY_OBJECT, "
             << "numhandle can only accept arrays with numerical elements"
         << std::ends;
      PyErr_SetString(PyExc_TypeError, stream.str().c_str());
      throw pybind11::error_already_set();
  }
  return;
}

std::string type2string(NPY_TYPES t_type){
  return kindstrings[t_type];
}

char type2char(NPY_TYPES t_type){
  return kindchars[t_type];
}

NPY_TYPES char2type(char e_type){
  return kindtypes[e_type];
}

template <class T>
inline std::string vector_str(const std::vector<T>& vec)
{
  std::ostringstream stream;
  stream << "(" << vec[0];

  for(std::size_t i = 1; i < vec.size(); i++){
    stream << ", " << vec[i];
  }
  stream << ")";
  return stream.str();
}

inline void check_size_match(std::vector<intp> dims, intp n)
{
  intp total = std::accumulate(dims.begin(),dims.end(),1,std::multiplies<intp>());
  if (total != n) {
    std::ostringstream stream;
    stream << "expected array size " << n
           << ", dimensions give array size " << total << std::ends;
    PyErr_SetString(PyExc_TypeError, stream.str().c_str());
    throw pybind11::error_already_set();
  }
  return;
}

} //namespace num_util
