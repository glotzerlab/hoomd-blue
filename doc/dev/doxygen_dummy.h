// \file prevent lint detector warning
// dummy file to trick doxygen into documenting shared pointers
namespace boost {
//! Shared pointer dummy
template<class T>
class shared_ptr
    {
    T *p; //!< Pointer
    }; }
