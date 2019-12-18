// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


/*! \file upp11_config.h
    \brief Helps unit tests setup the upp11 unit testing framework
    \details much of this is replicated from the old boost files, future unit
    tests are welcome to include only standard usage
    \note This file should be included only once and by a file that will
        compile into a unit test executable
*/

#include "hoomd/HOOMDMath.h"
#include "hoomd/HOOMDMPI.h"
#include "hoomd/ExecutionConfiguration.h"

#include "hoomd/extern/upp11/upp11.h"

#include <cmath>
#include <string>
#include <vector>

//! Macro to test if the difference between two floating-point values is within a tolerance
/*!
 * \param a First value to test
 * \param b Second value to test
 * \param eps Difference allowed between the two
 *
 * This assertion will pass if the difference between \a a and \a b is within a tolerance,
 * defined by \a eps times the smaller of the magnitude of \a a and \a b.
 *
 * \warning This assertion should not be used when one of the values should be zero. In that
 *          case, use UP_ASSERT_SMALL instead.
 */
#define UP_ASSERT_CLOSE(a,b,eps) \
upp11::TestCollection::getInstance().checkpoint(LOCATION, "UP_ASSERT_CLOSE"), \
upp11::TestAssert(LOCATION).assertTrue(std::abs((a)-(b)) <= (eps) * std::min(std::abs(a), std::abs(b)), \
                                       #a " (" + std::to_string(a) + ") close to " #b " (" + std::to_string(b) + ")")

//! Macro to test if a floating-point value is close to zero
/*!
 * \param a Value to test
 * \param eps Difference allowed from zero
 *
 * This assertion will pass if the absolute value of \a a is less than \a eps.
 */
#define UP_ASSERT_SMALL(a,eps) \
upp11::TestCollection::getInstance().checkpoint(LOCATION, "UP_ASSERT_SMALL"), \
upp11::TestAssert(LOCATION).assertTrue(std::abs(a) < (eps), #a " (" + std::to_string(a) + ") close to 0")

//! Macro to test if a value is greater than another
/*!
 * \param a First value to test
 * \param b Second value to test
 *
 * This assertion will pass if \a a > \b b.
 */
#define UP_ASSERT_GREATER(a,b) \
upp11::TestCollection::getInstance().checkpoint(LOCATION, "UP_ASSERT_GREATER"), \
upp11::TestAssert(LOCATION).assertTrue(a > b, #a " (" + std::to_string(a) + ") > " #b " (" + std::to_string(b) + ")")

//! Macro to test if a value is greater than or equal to another
/*!
 * \param a First value to test
 * \param b Second value to test
 *
 * This assertion will pass if \a a >= \b b.
 */
#define UP_ASSERT_GREATER_EQUAL(a,b) \
upp11::TestCollection::getInstance().checkpoint(LOCATION, "UP_ASSERT_GREATER_EQUAL"), \
upp11::TestAssert(LOCATION).assertTrue(a >= b, #a " (" + std::to_string(a) + ") >= " #b " (" + std::to_string(b) + ")")

//! Macro to test if a value is less than another
/*!
 * \param a First value to test
 * \param b Second value to test
 *
 * This assertion will pass if \a a < \b b.
 */
#define UP_ASSERT_LESS(a,b) \
upp11::TestCollection::getInstance().checkpoint(LOCATION, "UP_ASSERT_LESS"), \
upp11::TestAssert(LOCATION).assertTrue(a < b, #a " (" + std::to_string(a) + ") < " #b " (" + std::to_string(b) + ")")

//! Macro to test if a value is less than or equal to another
/*!
 * \param a First value to test
 * \param b Second value to test
 *
 * This assertion will pass if \a a <= \b b.
 */
#define UP_ASSERT_LESS_EQUAL(a,b) \
upp11::TestCollection::getInstance().checkpoint(LOCATION, "UP_ASSERT_LESS_EQUAL"), \
upp11::TestAssert(LOCATION).assertTrue(a <= b, #a " (" + std::to_string(a) + ") <= " #b " (" + std::to_string(b) + ")")

// ******** helper macros
#define CHECK_CLOSE(a,b,c) UP_ASSERT((std::abs((a)-(b)) <= ((c) * std::abs(a))) && (std::abs((a)-(b)) <= ((c) * std::abs(b))))
#define CHECK_SMALL(a,c) UP_ASSERT(std::abs(a) < c)
//! Helper macro for checking if two numbers are close
#define MY_CHECK_CLOSE(a,b,c) UP_ASSERT((std::abs((a)-(b)) <= ((c) * std::abs(a))) && (std::abs((a)-(b)) <= ((c) * std::abs(b))))
//! Helper macro for checking if a number is small
#define MY_CHECK_SMALL(a,c) CHECK_SMALL( a, Scalar(c))
//! Need a simple define for checking two values which are unsigned
#define CHECK_EQUAL_UINT(a,b) UP_ASSERT_EQUAL(a,(unsigned int)(b))

#define MY_ASSERT_EQUAL(a,b) UP_ASSERT(a == b)

//! Tolerance setting for near-zero comparisons
const Scalar tol_small = Scalar(1e-3);

//! Tolerance setting for comparisons
const Scalar tol = Scalar(1e-2);

//! Loose tolerance to be used with randomly generated and unpredictable comparisons
Scalar loose_tol = Scalar(10);

std::shared_ptr<ExecutionConfiguration> exec_conf_cpu;
std::shared_ptr<ExecutionConfiguration> exec_conf_gpu;

#ifdef ENABLE_MPI
#define HOOMD_UP_MAIN() \
int main(int argc, char **argv) \
    { \
    MPI_Init(&argc, &argv); \
    char dash_s[] = "-s"; \
    char zero[] = "0"; \
    std::vector<char *> new_argv(argv, argv+argc); \
    new_argv.push_back(dash_s); \
    new_argv.push_back(zero); \
    int val = upp11::TestMain().main(new_argv.size(), &new_argv[0]); \
    exec_conf_cpu.reset(); \
    exec_conf_gpu.reset(); \
    MPI_Finalize(); \
    return val; \
    }
#else
#define HOOMD_UP_MAIN() \
int main(int argc, char **argv) { \
    char dash_s[] = "-s"; \
    char zero[] = "0"; \
    std::vector<char *> new_argv(argv, argv+argc); \
    new_argv.push_back(dash_s); \
    new_argv.push_back(zero); \
    int val = upp11::TestMain().main(new_argv.size(), &new_argv[0]); \
    exec_conf_cpu.reset(); \
    exec_conf_gpu.reset(); \
    return val; \
}
#endif
