/*
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of NVIDIA CORPORATION nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// safe headers extracted from jitify.hpp

namespace jitify { namespace detail {

static const char* jitsafe_header_preinclude_h = R"(
// WAR for Thrust (which appears to have forgotten to include this in result_of_adaptable_function.h
#include <type_traits>

// WAR for Thrust (which appear to have forgotten to include this in error_code.h)
#include <string>

// WAR for Thrust (which only supports gnuc, clang or msvc)
#define __GNUC__ 4

// WAR for generics/shfl.h
#define THRUST_STATIC_ASSERT(x)

// WAR for CUB
#ifdef __host__
#undef __host__
#endif
#define __host__

// WAR to allow exceptions to be parsed
#define try
#define catch(...)
)";

static const char* jitsafe_header_float_h =
    "#pragma once\n"
    "\n"
    "inline __host__ __device__ float  jitify_int_as_float(int i)             "
    "{ union FloatInt { float f; int i; } fi; fi.i = i; return fi.f; }\n"
    "inline __host__ __device__ double jitify_longlong_as_double(long long i) "
    "{ union DoubleLongLong { double f; long long i; } fi; fi.i = i; return "
    "fi.f; }\n"
    "#define FLT_RADIX       2\n"
    "#define FLT_MANT_DIG    24\n"
    "#define DBL_MANT_DIG    53\n"
    "#define FLT_DIG         6\n"
    "#define DBL_DIG         15\n"
    "#define FLT_MIN_EXP     -125\n"
    "#define DBL_MIN_EXP     -1021\n"
    "#define FLT_MIN_10_EXP  -37\n"
    "#define DBL_MIN_10_EXP  -307\n"
    "#define FLT_MAX_EXP     128\n"
    "#define DBL_MAX_EXP     1024\n"
    "#define FLT_MAX_10_EXP  38\n"
    "#define DBL_MAX_10_EXP  308\n"
    "#define FLT_MAX         jitify_int_as_float(2139095039)\n"
    "#define DBL_MAX         jitify_longlong_as_double(9218868437227405311)\n"
    "#define FLT_EPSILON     jitify_int_as_float(872415232)\n"
    "#define DBL_EPSILON     jitify_longlong_as_double(4372995238176751616)\n"
    "#define FLT_MIN         jitify_int_as_float(8388608)\n"
    "#define DBL_MIN         jitify_longlong_as_double(4503599627370496)\n"
    "#define FLT_ROUNDS      1\n"
    "#if defined __cplusplus && __cplusplus >= 201103L\n"
    "#define FLT_EVAL_METHOD 0\n"
    "#define DECIMAL_DIG     21\n"
    "#endif\n";

static const char* jitsafe_header_limits_h =
    "#pragma once\n"
    "\n"
    "#if defined _WIN32 || defined _WIN64\n"
    " #define __WORDSIZE 32\n"
    "#else\n"
    " #if defined __x86_64__ && !defined __ILP32__\n"
    "  #define __WORDSIZE 64\n"
    " #else\n"
    "  #define __WORDSIZE 32\n"
    " #endif\n"
    "#endif\n"
    "#define MB_LEN_MAX  16\n"
    "#define CHAR_BIT    8\n"
    "#define SCHAR_MIN   (-128)\n"
    "#define SCHAR_MAX   127\n"
    "#define UCHAR_MAX   255\n"
    "#ifdef __CHAR_UNSIGNED__\n"
    " #define CHAR_MIN   0\n"
    " #define CHAR_MAX   UCHAR_MAX\n"
    "#else\n"
    " #define CHAR_MIN   SCHAR_MIN\n"
    " #define CHAR_MAX   SCHAR_MAX\n"
    "#endif\n"
    "#define SHRT_MIN    (-32768)\n"
    "#define SHRT_MAX    32767\n"
    "#define USHRT_MAX   65535\n"
    "#define INT_MIN     (-INT_MAX - 1)\n"
    "#define INT_MAX     2147483647\n"
    "#define UINT_MAX    4294967295U\n"
    "#if __WORDSIZE == 64\n"
    " # define LONG_MAX  9223372036854775807L\n"
    "#else\n"
    " # define LONG_MAX  2147483647L\n"
    "#endif\n"
    "#define LONG_MIN    (-LONG_MAX - 1L)\n"
    "#if __WORDSIZE == 64\n"
    " #define ULONG_MAX  18446744073709551615UL\n"
    "#else\n"
    " #define ULONG_MAX  4294967295UL\n"
    "#endif\n"
    "#define LLONG_MAX  9223372036854775807LL\n"
    "#define LLONG_MIN  (-LLONG_MAX - 1LL)\n"
    "#define ULLONG_MAX 18446744073709551615ULL\n";

static const char* jitsafe_header_iterator =
    "#pragma once\n"
    "\n"
    "namespace __jitify_iterator_ns {\n"
    "struct output_iterator_tag {};\n"
    "struct input_iterator_tag {};\n"
    "struct forward_iterator_tag {};\n"
    "struct bidirectional_iterator_tag {};\n"
    "struct random_access_iterator_tag {};\n"
    "template<class Iterator>\n"
    "struct iterator_traits {\n"
    "  typedef typename Iterator::iterator_category iterator_category;\n"
    "  typedef typename Iterator::value_type        value_type;\n"
    "  typedef typename Iterator::difference_type   difference_type;\n"
    "  typedef typename Iterator::pointer           pointer;\n"
    "  typedef typename Iterator::reference         reference;\n"
    "};\n"
    "template<class T>\n"
    "struct iterator_traits<T*> {\n"
    "  typedef random_access_iterator_tag iterator_category;\n"
    "  typedef T                          value_type;\n"
    "  typedef ptrdiff_t                  difference_type;\n"
    "  typedef T*                         pointer;\n"
    "  typedef T&                         reference;\n"
    "};\n"
    "template<class T>\n"
    "struct iterator_traits<T const*> {\n"
    "  typedef random_access_iterator_tag iterator_category;\n"
    "  typedef T                          value_type;\n"
    "  typedef ptrdiff_t                  difference_type;\n"
    "  typedef T const*                   pointer;\n"
    "  typedef T const&                   reference;\n"
    "};\n"
    "} // namespace __jitify_iterator_ns\n"
    "namespace std { using namespace __jitify_iterator_ns; }\n"
    "using namespace __jitify_iterator_ns;\n";

// TODO: This is incomplete; need floating point limits
static const char* jitsafe_header_limits =
    "#pragma once\n"
    "#include <climits>\n"
    "\n"
    "namespace __jitify_limits_ns {\n"
    "// TODO: Floating-point limits\n"
    "namespace __jitify_detail {\n"
    "template<class T, T Min, T Max, int Digits=-1>\n"
    "struct IntegerLimits {\n"
    "	static inline __host__ __device__ T min() { return Min; }\n"
    "	static inline __host__ __device__ T max() { return Max; }\n"
    "	enum {\n"
    "       is_specialized = true,\n"
    "		digits     = (Digits == -1) ? (int)(sizeof(T)*8 - (Min != 0)) "
    ": Digits,\n"
    "		digits10   = (digits * 30103) / 100000,\n"
    "		is_signed  = ((T)(-1)<0),\n"
    "		is_integer = true,\n"
    "		is_exact   = true,\n"
    "		radix      = 2,\n"
    "		is_bounded = true,\n"
    "		is_modulo  = false\n"
    "	};\n"
    "};\n"
    "} // namespace detail\n"
    "template<typename T> struct numeric_limits {\n"
    "    enum { is_specialized = false };\n"
    "};\n"
    "template<> struct numeric_limits<bool>               : public "
    "__jitify_detail::IntegerLimits<bool,              false,    true,1> {};\n"
    "template<> struct numeric_limits<char>               : public "
    "__jitify_detail::IntegerLimits<char,              CHAR_MIN, CHAR_MAX> "
    "{};\n"
    "template<> struct numeric_limits<signed char>        : public "
    "__jitify_detail::IntegerLimits<signed char,       SCHAR_MIN,SCHAR_MAX> "
    "{};\n"
    "template<> struct numeric_limits<unsigned char>      : public "
    "__jitify_detail::IntegerLimits<unsigned char,     0,        UCHAR_MAX> "
    "{};\n"
    "template<> struct numeric_limits<wchar_t>            : public "
    "__jitify_detail::IntegerLimits<wchar_t,           INT_MIN,  INT_MAX> {};\n"
    "template<> struct numeric_limits<short>              : public "
    "__jitify_detail::IntegerLimits<short,             SHRT_MIN, SHRT_MAX> "
    "{};\n"
    "template<> struct numeric_limits<unsigned short>     : public "
    "__jitify_detail::IntegerLimits<unsigned short,    0,        USHRT_MAX> "
    "{};\n"
    "template<> struct numeric_limits<int>                : public "
    "__jitify_detail::IntegerLimits<int,               INT_MIN,  INT_MAX> {};\n"
    "template<> struct numeric_limits<unsigned int>       : public "
    "__jitify_detail::IntegerLimits<unsigned int,      0,        UINT_MAX> "
    "{};\n"
    "template<> struct numeric_limits<long>               : public "
    "__jitify_detail::IntegerLimits<long,              LONG_MIN, LONG_MAX> "
    "{};\n"
    "template<> struct numeric_limits<unsigned long>      : public "
    "__jitify_detail::IntegerLimits<unsigned long,     0,        ULONG_MAX> "
    "{};\n"
    "template<> struct numeric_limits<long long>          : public "
    "__jitify_detail::IntegerLimits<long long,         LLONG_MIN,LLONG_MAX> "
    "{};\n"
    "template<> struct numeric_limits<unsigned long long> : public "
    "__jitify_detail::IntegerLimits<unsigned long long,0,        ULLONG_MAX> "
    "{};\n"
    "//template<typename T> struct numeric_limits { static const bool "
    "is_signed = ((T)(-1)<0); };\n"
    "} // namespace __jitify_limits_ns\n"
    "namespace std { using namespace __jitify_limits_ns; }\n"
    "using namespace __jitify_limits_ns;\n";

// TODO: This is highly incomplete
static const char* jitsafe_header_type_traits = R"(
    #pragma once
    #if __cplusplus >= 201103L
    namespace __jitify_type_traits_ns {

    template<bool B, class T = void> struct enable_if {};
    template<class T>                struct enable_if<true, T> { typedef T type; };
    #if __cplusplus >= 201402L
    template< bool B, class T = void > using enable_if_t = typename enable_if<B,T>::type;
    #endif

    struct true_type  {
      enum { value = true };
      operator bool() const { return true; }
    };
    struct false_type {
      enum { value = false };
      operator bool() const { return false; }
    };

    template<typename T> struct is_floating_point    : false_type {};
    template<> struct is_floating_point<float>       :  true_type {};
    template<> struct is_floating_point<double>      :  true_type {};
    template<> struct is_floating_point<long double> :  true_type {};

    template<class T> struct is_integral              : false_type {};
    template<> struct is_integral<bool>               :  true_type {};
    template<> struct is_integral<char>               :  true_type {};
    template<> struct is_integral<signed char>        :  true_type {};
    template<> struct is_integral<unsigned char>      :  true_type {};
    template<> struct is_integral<short>              :  true_type {};
    template<> struct is_integral<unsigned short>     :  true_type {};
    template<> struct is_integral<int>                :  true_type {};
    template<> struct is_integral<unsigned int>       :  true_type {};
    template<> struct is_integral<long>               :  true_type {};
    template<> struct is_integral<unsigned long>      :  true_type {};
    template<> struct is_integral<long long>          :  true_type {};
    template<> struct is_integral<unsigned long long> :  true_type {};

    template<typename T> struct is_signed    : false_type {};
    template<> struct is_signed<float>       :  true_type {};
    template<> struct is_signed<double>      :  true_type {};
    template<> struct is_signed<long double> :  true_type {};
    template<> struct is_signed<signed char> :  true_type {};
    template<> struct is_signed<short>       :  true_type {};
    template<> struct is_signed<int>         :  true_type {};
    template<> struct is_signed<long>        :  true_type {};
    template<> struct is_signed<long long>   :  true_type {};

    template<typename T> struct is_unsigned             : false_type {};
    template<> struct is_unsigned<unsigned char>      :  true_type {};
    template<> struct is_unsigned<unsigned short>     :  true_type {};
    template<> struct is_unsigned<unsigned int>       :  true_type {};
    template<> struct is_unsigned<unsigned long>      :  true_type {};
    template<> struct is_unsigned<unsigned long long> :  true_type {};

    template<typename T, typename U> struct is_same      : false_type {};
    template<typename T>             struct is_same<T,T> :  true_type {};

    template<class T> struct is_array : false_type {};
    template<class T> struct is_array<T[]> : true_type {};
    template<class T, size_t N> struct is_array<T[N]> : true_type {};

    //partial implementation only of is_function
    template<class> struct is_function : false_type { };
    template<class Ret, class... Args> struct is_function<Ret(Args...)> : true_type {}; //regular
    template<class Ret, class... Args> struct is_function<Ret(Args......)> : true_type {}; // variadic

    template<class> struct result_of;
    template<class F, typename... Args>
    struct result_of<F(Args...)> {
    // TODO: This is a hack; a proper implem is quite complicated.
    typedef typename F::result_type type;
    };

    template <class T> struct remove_reference { typedef T type; };
    template <class T> struct remove_reference<T&> { typedef T type; };
    template <class T> struct remove_reference<T&&> { typedef T type; };
    #if __cplusplus >= 201402L
    template< class T > using remove_reference_t = typename remove_reference<T>::type;
    #endif

    template<class T> struct remove_extent { typedef T type; };
    template<class T> struct remove_extent<T[]> { typedef T type; };
    template<class T, size_t N> struct remove_extent<T[N]> { typedef T type; };
    #if __cplusplus >= 201402L
    template< class T > using remove_extent_t = typename remove_extent<T>::type;
    #endif

    template< class T > struct remove_const          { typedef T type; };
    template< class T > struct remove_const<const T> { typedef T type; };
    template< class T > struct remove_volatile             { typedef T type; };
    template< class T > struct remove_volatile<volatile T> { typedef T type; };
    template< class T > struct remove_cv { typedef typename remove_volatile<typename remove_const<T>::type>::type type; };
    #if __cplusplus >= 201402L
    template< class T > using remove_cv_t       = typename remove_cv<T>::type;
    template< class T > using remove_const_t    = typename remove_const<T>::type;
    template< class T > using remove_volatile_t = typename remove_volatile<T>::type;
    #endif

    template<bool B, class T, class F> struct conditional { typedef T type; };
    template<class T, class F> struct conditional<false, T, F> { typedef F type; };
    #if __cplusplus >= 201402L
    template< bool B, class T, class F > using conditional_t = typename conditional<B,T,F>::type;
    #endif

    namespace __jitify_detail {
    template< class T, bool is_function_type = false > struct add_pointer { using type = typename remove_reference<T>::type*; };
    template< class T > struct add_pointer<T, true> { using type = T; };
    template< class T, class... Args > struct add_pointer<T(Args...), true> { using type = T(*)(Args...); };
    template< class T, class... Args > struct add_pointer<T(Args..., ...), true> { using type = T(*)(Args..., ...); };
    }
    template< class T > struct add_pointer : __jitify_detail::add_pointer<T, is_function<T>::value> {};
    #if __cplusplus >= 201402L
    template< class T > using add_pointer_t = typename add_pointer<T>::type;
    #endif

    template< class T > struct decay {
    private:
      typedef typename remove_reference<T>::type U;
    public:
      typedef typename conditional<is_array<U>::value, typename remove_extent<U>::type*,
        typename conditional<is_function<U>::value,typename add_pointer<U>::type,typename remove_cv<U>::type
        >::type>::type type;
    };
    #if __cplusplus >= 201402L
    template< class T > using decay_t = typename decay<T>::type;
    #endif

    } // namespace __jtiify_type_traits_ns
    namespace std { using namespace __jitify_type_traits_ns; }
    using namespace __jitify_type_traits_ns;
    #endif // c++11
)";

// TODO: INT_FAST8_MAX et al. and a few other misc constants
static const char* jitsafe_header_stdint_h =
    "#pragma once\n"
    "#include <climits>\n"
    "namespace __jitify_stdint_ns {\n"
    "typedef signed char      int8_t;\n"
    "typedef signed short     int16_t;\n"
    "typedef signed int       int32_t;\n"
    "typedef signed long long int64_t;\n"
    "typedef signed char      int_fast8_t;\n"
    "typedef signed short     int_fast16_t;\n"
    "typedef signed int       int_fast32_t;\n"
    "typedef signed long long int_fast64_t;\n"
    "typedef signed char      int_least8_t;\n"
    "typedef signed short     int_least16_t;\n"
    "typedef signed int       int_least32_t;\n"
    "typedef signed long long int_least64_t;\n"
    "typedef signed long long intmax_t;\n"
    "typedef signed long      intptr_t; //optional\n"
    "typedef unsigned char      uint8_t;\n"
    "typedef unsigned short     uint16_t;\n"
    "typedef unsigned int       uint32_t;\n"
    "typedef unsigned long long uint64_t;\n"
    "typedef unsigned char      uint_fast8_t;\n"
    "typedef unsigned short     uint_fast16_t;\n"
    "typedef unsigned int       uint_fast32_t;\n"
    "typedef unsigned long long uint_fast64_t;\n"
    "typedef unsigned char      uint_least8_t;\n"
    "typedef unsigned short     uint_least16_t;\n"
    "typedef unsigned int       uint_least32_t;\n"
    "typedef unsigned long long uint_least64_t;\n"
    "typedef unsigned long long uintmax_t;\n"
    "typedef unsigned long      uintptr_t; //optional\n"
    "#define INT8_MIN    SCHAR_MIN\n"
    "#define INT16_MIN   SHRT_MIN\n"
    "#define INT32_MIN   INT_MIN\n"
    "#define INT64_MIN   LLONG_MIN\n"
    "#define INT8_MAX    SCHAR_MAX\n"
    "#define INT16_MAX   SHRT_MAX\n"
    "#define INT32_MAX   INT_MAX\n"
    "#define INT64_MAX   LLONG_MAX\n"
    "#define UINT8_MAX   UCHAR_MAX\n"
    "#define UINT16_MAX  USHRT_MAX\n"
    "#define UINT32_MAX  UINT_MAX\n"
    "#define UINT64_MAX  ULLONG_MAX\n"
    "#define INTPTR_MIN  LONG_MIN\n"
    "#define INTMAX_MIN  LLONG_MIN\n"
    "#define INTPTR_MAX  LONG_MAX\n"
    "#define INTMAX_MAX  LLONG_MAX\n"
    "#define UINTPTR_MAX ULONG_MAX\n"
    "#define UINTMAX_MAX ULLONG_MAX\n"
    "#define PTRDIFF_MIN INTPTR_MIN\n"
    "#define PTRDIFF_MAX INTPTR_MAX\n"
    "#define SIZE_MAX    UINT64_MAX\n"
    "} // namespace __jitify_stdint_ns\n"
    "namespace std { using namespace __jitify_stdint_ns; }\n"
    "using namespace __jitify_stdint_ns;\n";

// TODO: offsetof
static const char* jitsafe_header_stddef_h =
    "#pragma once\n"
    "#include <climits>\n"
    "namespace __jitify_stddef_ns {\n"
    //"enum { NULL = 0 };\n"
    "typedef unsigned long size_t;\n"
    "typedef   signed long ptrdiff_t;\n"
    "} // namespace __jitify_stddef_ns\n"
    "namespace std { using namespace __jitify_stddef_ns; }\n"
    "using namespace __jitify_stddef_ns;\n";

static const char* jitsafe_header_stdlib_h =
    "#pragma once\n"
    "#include <stddef.h>\n";
static const char* jitsafe_header_stdio_h =
    "#pragma once\n"
    "#include <stddef.h>\n"
    "#define FILE int\n"
    "int fflush ( FILE * stream );\n"
    "int fprintf ( FILE * stream, const char * format, ... );\n";

static const char* jitsafe_header_string_h =
    "#pragma once\n"
    "char* strcpy ( char * destination, const char * source );\n"
    "int strcmp ( const char * str1, const char * str2 );\n"
    "char* strerror( int errnum );\n";

static const char* jitsafe_header_cstring =
    "#pragma once\n"
    "\n"
    "namespace __jitify_cstring_ns {\n"
    "char* strcpy ( char * destination, const char * source );\n"
    "int strcmp ( const char * str1, const char * str2 );\n"
    "char* strerror( int errnum );\n"
    "} // namespace __jitify_cstring_ns\n"
    "namespace std { using namespace __jitify_cstring_ns; }\n"
    "using namespace __jitify_cstring_ns;\n";

// HACK TESTING (WAR for cub)
static const char* jitsafe_header_iostream =
    "#pragma once\n"
    "#include <ostream>\n"
    "#include <istream>\n";
// HACK TESTING (WAR for Thrust)
static const char* jitsafe_header_ostream =
    "#pragma once\n"
    "\n"
    "namespace __jitify_ostream_ns {\n"
    "template<class CharT,class Traits=void>\n"  // = std::char_traits<CharT>
                                                 // >\n"
    "struct basic_ostream {\n"
    "};\n"
    "typedef basic_ostream<char> ostream;\n"
    "ostream& endl(ostream& os);\n"
    "ostream& operator<<( ostream&, ostream& (*f)( ostream& ) );\n"
    "template< class CharT, class Traits > basic_ostream<CharT, Traits>& endl( "
    "basic_ostream<CharT, Traits>& os );\n"
    "template< class CharT, class Traits > basic_ostream<CharT, Traits>& "
    "operator<<( basic_ostream<CharT,Traits>& os, const char* c );\n"
    "#if __cplusplus >= 201103L\n"
    "template< class CharT, class Traits, class T > basic_ostream<CharT, "
    "Traits>& operator<<( basic_ostream<CharT,Traits>&& os, const T& value );\n"
    "#endif  // __cplusplus >= 201103L\n"
    "} // namespace __jitify_ostream_ns\n"
    "namespace std { using namespace __jitify_ostream_ns; }\n"
    "using namespace __jitify_ostream_ns;\n";

static const char* jitsafe_header_istream =
    "#pragma once\n"
    "\n"
    "namespace __jitify_istream_ns {\n"
    "template<class CharT,class Traits=void>\n"  // = std::char_traits<CharT>
                                                 // >\n"
    "struct basic_istream {\n"
    "};\n"
    "typedef basic_istream<char> istream;\n"
    "} // namespace __jitify_istream_ns\n"
    "namespace std { using namespace __jitify_istream_ns; }\n"
    "using namespace __jitify_istream_ns;\n";

static const char* jitsafe_header_sstream =
    "#pragma once\n"
    "#include <ostream>\n"
    "#include <istream>\n";

static const char* jitsafe_header_utility =
    "#pragma once\n"
    "namespace __jitify_utility_ns {\n"
    "template<class T1, class T2>\n"
    "struct pair {\n"
    "	T1 first;\n"
    "	T2 second;\n"
    "	inline pair() {}\n"
    "	inline pair(T1 const& first_, T2 const& second_)\n"
    "		: first(first_), second(second_) {}\n"
    "	// TODO: Standard includes many more constructors...\n"
    "	// TODO: Comparison operators\n"
    "};\n"
    "template<class T1, class T2>\n"
    "pair<T1,T2> make_pair(T1 const& first, T2 const& second) {\n"
    "	return pair<T1,T2>(first, second);\n"
    "}\n"
    "} // namespace __jitify_utility_ns\n"
    "namespace std { using namespace __jitify_utility_ns; }\n"
    "using namespace __jitify_utility_ns;\n";

// TODO: incomplete
static const char* jitsafe_header_vector =
    "#pragma once\n"
    "namespace __jitify_vector_ns {\n"
    "template<class T, class Allocator=void>\n"  // = std::allocator> \n"
    "struct vector {\n"
    "};\n"
    "} // namespace __jitify_vector_ns\n"
    "namespace std { using namespace __jitify_vector_ns; }\n"
    "using namespace __jitify_vector_ns;\n";

// TODO: incomplete
static const char* jitsafe_header_string =
    "#pragma once\n"
    "namespace __jitify_string_ns {\n"
    "template<class CharT,class Traits=void,class Allocator=void>\n"
    "struct basic_string {\n"
    "basic_string();\n"
    "basic_string( const CharT* s );\n"  //, const Allocator& alloc =
                                         // Allocator() );\n"
    "const CharT* c_str() const;\n"
    "bool empty() const;\n"
    "void operator+=(const char *);\n"
    "void operator+=(const basic_string &);\n"
    "};\n"
    "typedef basic_string<char> string;\n"
    "} // namespace __jitify_string_ns\n"
    "namespace std { using namespace __jitify_string_ns; }\n"
    "using namespace __jitify_string_ns;\n";

// TODO: incomplete
static const char* jitsafe_header_stdexcept =
    "#pragma once\n"
    "namespace __jitify_stdexcept_ns {\n"
    "struct runtime_error {\n"
    "explicit runtime_error( const std::string& what_arg );"
    "explicit runtime_error( const char* what_arg );"
    "virtual const char* what() const;\n"
    "};\n"
    "} // namespace __jitify_stdexcept_ns\n"
    "namespace std { using namespace __jitify_stdexcept_ns; }\n"
    "using namespace __jitify_stdexcept_ns;\n";

// TODO: incomplete
static const char* jitsafe_header_complex =
    "#pragma once\n"
    "namespace __jitify_complex_ns {\n"
    "template<typename T>\n"
    "class complex {\n"
    "	T _real;\n"
    "	T _imag;\n"
    "public:\n"
    "	complex() : _real(0), _imag(0) {}\n"
    "	complex(T const& real, T const& imag)\n"
    "		: _real(real), _imag(imag) {}\n"
    "	complex(T const& real)\n"
    "               : _real(real), _imag(static_cast<T>(0)) {}\n"
    "	T const& real() const { return _real; }\n"
    "	T&       real()       { return _real; }\n"
    "	void real(const T &r) { _real = r; }\n"
    "	T const& imag() const { return _imag; }\n"
    "	T&       imag()       { return _imag; }\n"
    "	void imag(const T &i) { _imag = i; }\n"
    "       complex<T>& operator+=(const complex<T> z)\n"
    "         { _real += z.real(); _imag += z.imag(); return *this; }\n"
    "};\n"
    "template<typename T>\n"
    "complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs)\n"
    "  { return complex<T>(lhs.real()*rhs.real()-lhs.imag()*rhs.imag(),\n"
    "                      lhs.real()*rhs.imag()+lhs.imag()*rhs.real()); }\n"
    "template<typename T>\n"
    "complex<T> operator*(const complex<T>& lhs, const T & rhs)\n"
    "  { return complexs<T>(lhs.real()*rhs,lhs.imag()*rhs); }\n"
    "template<typename T>\n"
    "complex<T> operator*(const T& lhs, const complex<T>& rhs)\n"
    "  { return complexs<T>(rhs.real()*lhs,rhs.imag()*lhs); }\n"
    "} // namespace __jitify_complex_ns\n"
    "namespace std { using namespace __jitify_complex_ns; }\n"
    "using namespace __jitify_complex_ns;\n";

// TODO: This is incomplete (missing binary and integer funcs, macros,
// constants, types)
static const char* jitsafe_header_math =
    "#pragma once\n"
    "namespace __jitify_math_ns {\n"
    "#if __cplusplus >= 201103L\n"
    "#define DEFINE_MATH_UNARY_FUNC_WRAPPER(f) \\\n"
    "	inline double      f(double x)         { return ::f(x); } \\\n"
    "	inline float       f##f(float x)       { return ::f(x); } \\\n"
    "	/*inline long double f##l(long double x) { return ::f(x); }*/ \\\n"
    "	inline float       f(float x)          { return ::f(x); } \\\n"
    "	/*inline long double f(long double x)    { return ::f(x); }*/\n"
    "#else\n"
    "#define DEFINE_MATH_UNARY_FUNC_WRAPPER(f) \\\n"
    "	inline double      f(double x)         { return ::f(x); } \\\n"
    "	inline float       f##f(float x)       { return ::f(x); } \\\n"
    "	/*inline long double f##l(long double x) { return ::f(x); }*/\n"
    "#endif\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(cos)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(sin)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(tan)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(acos)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(asin)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(atan)\n"
    "template<typename T> inline T atan2(T y, T x) { return ::atan2(y, x); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(cosh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(sinh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(tanh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(exp)\n"
    "template<typename T> inline T frexp(T x, int* exp) { return ::frexp(x, "
    "exp); }\n"
    "template<typename T> inline T ldexp(T x, int  exp) { return ::ldexp(x, "
    "exp); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log10)\n"
    "template<typename T> inline T modf(T x, T* intpart) { return ::modf(x, "
    "intpart); }\n"
    "template<typename T> inline T pow(T x, T y) { return ::pow(x, y); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(sqrt)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(ceil)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(floor)\n"
    "template<typename T> inline T fmod(T n, T d) { return ::fmod(n, d); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(fabs)\n"
    "template<typename T> inline T abs(T x) { return ::abs(x); }\n"
    "#if __cplusplus >= 201103L\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(acosh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(asinh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(atanh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(exp2)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(expm1)\n"
    "template<typename T> inline int ilogb(T x) { return ::ilogb(x); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log1p)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log2)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(logb)\n"
    "template<typename T> inline T scalbn (T x, int n)  { return ::scalbn(x, "
    "n); }\n"
    "template<typename T> inline T scalbln(T x, long n) { return ::scalbn(x, "
    "n); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(cbrt)\n"
    "template<typename T> inline T hypot(T x, T y) { return ::hypot(x, y); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(erf)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(erfc)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(tgamma)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(lgamma)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(trunc)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(round)\n"
    "template<typename T> inline long lround(T x) { return ::lround(x); }\n"
    "template<typename T> inline long long llround(T x) { return ::llround(x); "
    "}\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(rint)\n"
    "template<typename T> inline long lrint(T x) { return ::lrint(x); }\n"
    "template<typename T> inline long long llrint(T x) { return ::llrint(x); "
    "}\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(nearbyint)\n"
    // TODO: remainder, remquo, copysign, nan, nextafter, nexttoward, fdim,
    // fmax, fmin, fma
    "#endif\n"
    "#undef DEFINE_MATH_UNARY_FUNC_WRAPPER\n"
    "} // namespace __jitify_math_ns\n"
    "namespace std { using namespace __jitify_math_ns; }\n"
    "#define M_PI 3.14159265358979323846\n"
    // Note: Global namespace already includes CUDA math funcs
    "//using namespace __jitify_math_ns;\n";

// TODO: incomplete
static const char* jitsafe_header_mutex = R"(
    #pragma once
    #if __cplusplus >= 201103L
    namespace __jitify_mutex_ns {
    class mutex {
    public:
    void lock();
    bool try_lock();
    void unlock();
    };
    } // namespace __jitify_mutex_ns
    namespace std { using namespace __jitify_mutex_ns; }
    using namespace __jitify_mutex_ns;
    #endif
 )";

static const char* jitsafe_header_algorithm = R"(
    #pragma once
    #if __cplusplus >= 201103L
    namespace __jitify_algorithm_ns {

    #if __cplusplus == 201103L
    #define JITIFY_CXX14_CONSTEXPR
    #else
    #define JITIFY_CXX14_CONSTEXPR constexpr
    #endif

    template<class T> JITIFY_CXX14_CONSTEXPR const T& max(const T& a, const T& b)
    {
      return (b > a) ? b : a;
    }
    template<class T> JITIFY_CXX14_CONSTEXPR const T& min(const T& a, const T& b)
    {
      return (b < a) ? b : a;
    }

    #endif
    } // namespace __jitify_algorithm_ns
    namespace std { using namespace __jitify_algorithm_ns; }
    using namespace __jitify_algorithm_ns;
    #endif
 )";

static const char* jitsafe_header_time_h = R"(
    #pragma once
    #define NULL 0
    #define CLOCKS_PER_SEC 1000000
    namespace __jitify_time_ns {
    typedef unsigned long size_t;
    typedef long clock_t;
    typedef long time_t;
    struct tm {
      int tm_sec;
      int tm_min;
      int tm_hour;
      int tm_mday;
      int tm_mon;
      int tm_year;
      int tm_wday;
      int tm_yday;
      int tm_isdst;
    };
    #if __cplusplus >= 201703L
    struct timespec {
      time_t tv_sec;
      long tv_nsec;
    };
    #endif
    }  // namespace __jitify_time_ns
    namespace std { using namespace __jitify_time_ns; }
    using namespace __jitify_time_ns;
 )";

static const char* jitsafe_headers[] = {
    jitsafe_header_preinclude_h, jitsafe_header_float_h,
    jitsafe_header_float_h,      jitsafe_header_limits_h,
    jitsafe_header_limits_h,     jitsafe_header_stdint_h,
    jitsafe_header_stdint_h,     jitsafe_header_stddef_h,
    jitsafe_header_stddef_h,     jitsafe_header_stdlib_h,
    jitsafe_header_stdlib_h,     jitsafe_header_stdio_h,
    jitsafe_header_stdio_h,      jitsafe_header_string_h,
    jitsafe_header_cstring,      jitsafe_header_iterator,
    jitsafe_header_limits,       jitsafe_header_type_traits,
    jitsafe_header_utility,      jitsafe_header_math,
    jitsafe_header_math,         jitsafe_header_complex,
    jitsafe_header_iostream,     jitsafe_header_ostream,
    jitsafe_header_istream,      jitsafe_header_sstream,
    jitsafe_header_vector,       jitsafe_header_string,
    jitsafe_header_stdexcept,    jitsafe_header_mutex,
    jitsafe_header_algorithm,    jitsafe_header_time_h,
    jitsafe_header_time_h};

}  // namespace detail

}  // namespace jitify
