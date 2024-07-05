// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file WarpTools.cuh
 * \brief Wrappers around CUB primitives for warp-level parallel primitives.
 */

#pragma once

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include <type_traits>

#ifndef __CUDACC_RTC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif

#if defined(__HIP_PLATFORM_HCC__)
#include <hipcub/hipcub.hpp>
#else
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>
#endif

#ifndef __CUDACC_RTC__
#pragma GCC diagnostic pop
#endif

#define DEVICE __device__ __forceinline__

namespace hoomd
    {
namespace detail
    {
    //! Computes warp-level reduction using shuffle instructions
    /*!
     * Reduction operations are performed at the warp or sub-warp level using shuffle instructions.
     * The sub-warp is defined as a consecutive group of threads that is (1) smaller than the
     * hardware warp size (32 threads) and (2) a power of 2. For additional details about any
     * operator, refer to the CUB documentation.
     *
     * This class is a thin wrapper around cub::WarpReduce. The CUB scan classes nominally request
     * "temporary" memory, which is shared memory for non-shuffle scans. However, the shuffle-based
     * scan does not use any shared memory, and so this temporary variable is put unused into a
     * register. The compiler can then optimize this out. We explicitly ensure that the storage type
     * is an empty date type.
     *
     * \tparam T data type to scan
     * \tparam LOGICAL_WARP_THREADS number of threads in a "logical" warp, must be a multiple of 2.
     * \tparam PTX_ARCH PTX architecture to build for, must be at least 300 (Kepler).
     */

#ifdef __HIP_PLATFORM_HCC__
template<typename T, int LOGICAL_WARP_THREADS = HIPCUB_WARP_THREADS, int PTX_ARCH = HIPCUB_ARCH>
#else
template<typename T, int LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS, int PTX_ARCH = CUB_PTX_ARCH>
#endif
class WarpReduce
    {
    public:
    DEVICE WarpReduce()
        {
#ifdef __HIP_PLATFORM_NVCC__
        static_assert(PTX_ARCH >= 300, "PTX architecture must be >= 300");
        static_assert(LOGICAL_WARP_THREADS <= CUB_PTX_WARP_THREADS,
                      "Logical warp size cannot exceed hardware warp size");
#else
        static_assert(LOGICAL_WARP_THREADS <= HIPCUB_WARP_THREADS,
                      "Logical warp size cannot exceed hardware warp size");
#endif
        static_assert(LOGICAL_WARP_THREADS && !(LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1)),
                      "Logical warp size must be a power of 2");
        }

    //! Sum reduction.
    /*!
     * \param input Thread data to sum.
     * \returns output The result of the sum reduction in thread 0 of the (sub-)warp.
     *
     * The sum reduction for a 4-thread sub-warp with \a input <tt>{1,2,3,4}</tt> gives \a output
     * <tt>10</tt> in thread 0. The result in the other threads is undefined.
     */
    DEVICE T Sum(T input)
        {
#ifdef __HIP_PLATFORM_HCC__
        return Reduce(input, hipcub::Sum());
#else
        return Reduce(input, cub::Sum());
#endif
        }

    //! Sum reduction over valid items.
    /*!
     * \param input Thread data to sum.
     * \param valid_items Total number of valid items in the (sub)-warp.
     * \returns output The result of the sum reduction in thread 0 of the (sub-)warp.
     *
     * The number of valid items may be smaller than the (sub-)warp. For example, if \a valid items
     * is 3, then the sum reduction for a 4-thread sub-warp with \a input <tt>{1,2,3,4}</tt> gives
     * \a output <tt>6</tt> in thread 0. The result in the other threads is undefined.
     */
    DEVICE T Sum(T input, int valid_items)
        {
#ifdef __HIP_PLATFORM_HCC__
        return Reduce(input, hipcub::Sum(), valid_items);
#else
        return Reduce(input, cub::Sum(), valid_items);
#endif
        }

    //! Custom reduction.
    /*!
     * \param input Thread data to sum.
     * \param reduce_op Custom reduction operation.
     * \returns output The result of the reduction in thread 0 of the (sub-)warp.
     *
     * \tparam ReduceOpT The type of the reduction operation.
     *
     * This is a generalization of Sum() to custom operators.
     */
    template<typename ReduceOpT> DEVICE T Reduce(T input, ReduceOpT reduce_op)
        {
        // shuffle-based reduce does not need temporary space, so we let the compiler optimize this
        // dummy variable out
        TempStorage tmp;
        return MyWarpReduce(tmp).Reduce(input, reduce_op);
        }

    //! Custom reduction over valid items.
    /*!
     * \param input Thread data to sum.
     * \param reduce_op Custom reduction operation.
     * \param valid_items Total number of valid items in the (sub)-warp.
     * \returns output The result of the reduction in thread 0 of the (sub-)warp.
     *
     * \tparam ReduceOpT The type of the reduction operation.
     *
     * This is a generalization of Sum() over valid items to custom operators.
     */
    template<typename ReduceOpT> DEVICE T Reduce(T input, ReduceOpT reduce_op, int valid_items)
        {
        // shuffle-based reduce does not need temporary space, so we let the compiler optimize this
        // dummy variable out
        TempStorage tmp;
        return MyWarpReduce(tmp).Reduce(input, reduce_op, valid_items);
        }

    private:
#ifdef __HIP_PLATFORM_HCC__
    typedef hipcub::WarpReduce<T, LOGICAL_WARP_THREADS, PTX_ARCH>
        MyWarpReduce; //!< CUB shuffle-based reduce
#else
    typedef cub::WarpReduce<T, LOGICAL_WARP_THREADS, PTX_ARCH>
        MyWarpReduce; //!< CUB shuffle-based reduce
#endif
    typedef typename MyWarpReduce::TempStorage
        TempStorage; //!< Nominal data type for CUB temporary storage

#ifdef __HIP_PLATFORM_HCC__
    static_assert(std::is_empty<TempStorage>::value, "WarpReduce requires temp storage ");
#else
// we would like to make a similar guarantee with NVIDA CUB too, but TempStorage is not an empty
// type even if it internally uses WarpReduceShfl
// static_assert(std::is_empty<TempStorage>::value, "WarpReduce requires temp storage ");
#endif
    };

//! Computes warp-level scan (prefix sum) using shuffle instructions
/*!
 * Scan operations are performed at the warp or sub-warp level using shuffle instructions. The
 * sub-warp is defined as a consecutive group of threads that is (1) smaller than the hardware warp
 * size (32 threads) and (2) a power of 2. For additional details about any operator, refer to the
 * CUB documentation.
 *
 * This class is a thin wrapper around hipcub::WarpScan. The CUB scan classes nominally request
 * "temporary" memory, which is shared memory for non-shuffle scans. However, the shuffle-based scan
 * does not use any shared memory, and so this temporary variable is put unused into a register. The
 * compiler can then optimize this out. Care must be taken to monitor the CUB implementation in
 * future to ensure the temporary memory is never used.
 *
 * \tparam T data type to scan
 * \tparam LOGICAL_WARP_THREADS number of threads in a "logical" warp, must be a multiple of 2.
 * \tparam PTX_ARCH PTX architecture to build for, must be at least 300 (Kepler).
 */
#ifdef __HIP_PLATFORM_HCC__
template<typename T, int LOGICAL_WARP_THREADS = HIPCUB_WARP_THREADS, int PTX_ARCH = HIPCUB_ARCH>
#else
template<typename T, int LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS, int PTX_ARCH = CUB_PTX_ARCH>
#endif
class WarpScan
    {
    public:
    DEVICE WarpScan()
        {
#ifdef __HIP_PLATFORM_NVCC__
        static_assert(PTX_ARCH >= 300, "PTX architecture must be >= 300");
        static_assert(LOGICAL_WARP_THREADS <= CUB_PTX_WARP_THREADS,
                      "Logical warp size cannot exceed hardware warp size");
#else
        static_assert(LOGICAL_WARP_THREADS <= HIPCUB_WARP_THREADS,
                      "Logical warp size cannot exceed hardware warp size");
#endif
        static_assert(LOGICAL_WARP_THREADS && !(LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1)),
                      "Logical warp size must be a power of 2");
        }

    //! Inclusive sum for each thread in logical warp.
    /*!
     * \param input Thread data to sum.
     * \param output Result of scan for this thread.
     *
     * The inclusive sum for a 4-thread sub-warp with \a input <tt>{1,2,3,4}</tt> gives \a output
     * <tt>{1,3,6,10}</tt>.
     */
    DEVICE void InclusiveSum(T input, T& output)
        {
#ifdef __HIP_PLATFORM_HCC__
        InclusiveScan(input, output, hipcub::Sum());
#else
        InclusiveScan(input, output, cub::Sum());
#endif
        }

    //! Inclusive sum for each thread in logical warp, plus accumulation for all.
    /*!
     * \param input Thread data to sum.
     * \param output Result of scan for this thread.
     * \param aggregate Total sum of all threads.
     *
     * The inclusive sum for a 4-thread sub-warp with \a input <tt>{1,2,3,4}</tt> gives \a output
     * <tt>{1,3,6,10}</tt>, and the \a aggregate is 10 for all threads in the sub-warp.
     */
    DEVICE void InclusiveSum(T input, T& output, T& aggregate)
        {
#ifdef __HIP_PLATFORM_HCC__
        InclusiveScan(input, output, hipcub::Sum(), aggregate);
#else
        InclusiveScan(input, output, cub::Sum(), aggregate);
#endif
        }

    //! Inclusive scan with a custom scan operator.
    /*!
     * \param input Thread data to sum.
     * \param output Result of scan for this thread.
     * \param scan_op Binary scan operator.
     *
     * This operator is equivalent to InclusiveSum if \a scan_op were hipcub::Sum().
     *
     * \tparam ScanOpT <b>inferred</b> Binary scan operator type.
     */
    template<class ScanOpT> DEVICE void InclusiveScan(T input, T& output, ScanOpT scan_op)
        {
        // shuffle-based scan does not need temporary space, so we let the compiler optimize this
        // dummy variable out
        TempStorage tmp;
        MyWarpScan(tmp).InclusiveScan(input, output, scan_op);
        }

    //! Inclusive scan with a custom scan operator, plus accumulation for all.
    /*!
     * \param input Thread data to sum.
     * \param output Result of scan for this thread.
     * \param scan_op Binary scan operator.
     * \param aggregate Total scan of all threads.
     *
     * This operator is equivalent to InclusiveSum if \a scan_op were hipcub::Sum().
     *
     * \tparam ScanOpT <b>inferred</b> Binary scan operator type.
     */
    template<class ScanOpT>
    DEVICE void InclusiveScan(T input, T& output, ScanOpT scan_op, T& aggregate)
        {
        // shuffle-based scan does not need temporary space, so we let the compiler optimize this
        // dummy variable out
        TempStorage tmp;
        MyWarpScan(tmp).InclusiveScan(input, output, scan_op, aggregate);
        }

    //! Exclusive sum for each thread in logical warp.
    /*!
     * \param input Thread data to sum.
     * \param output Result of scan for this thread.
     *
     * The inclusive sum for a 4-thread sub-warp with \a input <tt>{1,2,3,4}</tt> gives \a output
     * <tt>{0,1,3,6}</tt>. (The first thread is initialized from a value of zero.)
     */
    DEVICE void ExclusiveSum(T input, T& output)
        {
        T initial = 0;
#ifdef __HIP_PLATFORM_HCC__
        ExclusiveScan(input, output, initial, hipcub::Sum());
#else
        ExclusiveScan(input, output, initial, cub::Sum());
#endif
        }

    //! Exclusive sum for each thread in logical warp, plus accumulation for all.
    /*!
     * \param input Thread data to sum.
     * \param output Result of scan for this thread.
     * \param aggregate Total sum of all threads.
     *
     * The inclusive sum for a 4-thread sub-warp with \a input <tt>{1,2,3,4}</tt> gives \a output
     * <tt>{0,1,3,6}</tt>, and the \a aggregate is 10 for all threads in the sub-warp.
     */
    DEVICE void ExclusiveSum(T input, T& output, T& aggregate)
        {
        T initial = 0;
#ifdef __HIP_PLATFORM_HCC__
        ExclusiveScan(input, output, initial, hipcub::Sum(), aggregate);
#else
        ExclusiveScan(input, output, initial, cub::Sum(), aggregate);
#endif
        }

    //! Exclusive scan with a custom scan operator.
    /*!
     * \param input Thread data to sum.
     * \param output Result of scan for this thread.
     * \param scan_op Binary scan operator.
     *
     * This operator is equivalent to ExclusiveSum if \a scan_op were hipcub::Sum().
     *
     * \tparam ScanOpT <b>inferred</b> Binary scan operator type.
     */
    template<class ScanOpT> DEVICE void ExclusiveScan(T input, T& output, ScanOpT scan_op)
        {
        // shuffle-based scan does not need temporary space, so we let the compiler optimize this
        // dummy variable out
        TempStorage tmp;
        MyWarpScan scan(tmp);
        scan.ExclusiveScan(input, output, scan_op);
        }

    //! Exclusive scan with a custom scan operator and initial value.
    /*!
     * \param input Thread data to sum.
     * \param output Result of scan for this thread.
     * \param initial Initial value for exclusive sum within logical warp.
     * \param scan_op Binary scan operator.
     *
     * This operator is equivalent to ExclusiveSum if \a scan_op were hipcub::Sum() and \a initial
     * were zero.
     *
     * \tparam ScanOpT <b>inferred</b> Binary scan operator type.
     */
    template<class ScanOpT>
    DEVICE void ExclusiveScan(T input, T& output, T initial, ScanOpT scan_op)
        {
        // shuffle-based scan does not need temporary space, so we let the compiler optimize this
        // dummy variable out
        TempStorage tmp;
        MyWarpScan scan(tmp);
        scan.ExclusiveScan(input, output, initial, scan_op);
        }

    //! Exclusive scan with a custom scan operator, plus accumulation for all.
    /*!
     * \param input Thread data to sum.
     * \param output Result of scan for this thread.
     * \param scan_op Binary scan operator.
     * \param aggregate Total scan of all threads.
     *
     * This operator is equivalent to ExclusiveSum if \a scan_op were hipcub::Sum().
     *
     * \tparam ScanOpT <b>inferred</b> Binary scan operator type.
     */
    template<class ScanOpT>
    DEVICE void ExclusiveScan(T input, T& output, ScanOpT scan_op, T& aggregate)
        {
        // shuffle-based scan does not need temporary space, so we let the compiler optimize this
        // dummy variable out
        TempStorage tmp;
        MyWarpScan scan(tmp);
        scan.ExclusiveScan(input, output, scan_op, aggregate);
        }

    //! Exclusive scan with a custom scan operator and initial value, plus accumulation for all.
    /*!
     * \param input Thread data to sum.
     * \param output Result of scan for this thread.
     * \param initial Initial value for exclusive sum within logical warp.
     * \param scan_op Binary scan operator.
     * \param aggregate Total scan of all threads.
     *
     * This operator is equivalent to ExclusiveSum if \a scan_op were hipcub::Sum() and \a initial
     * were zero.
     *
     * \tparam ScanOpT <b>inferred</b> Binary scan operator type.
     */
    template<class ScanOpT>
    DEVICE void ExclusiveScan(T input, T& output, T initial, ScanOpT scan_op, T& aggregate)
        {
        // shuffle-based scan does not need temporary space, so we let the compiler optimize this
        // dummy variable out
        TempStorage tmp;
        MyWarpScan scan(tmp);
        scan.ExclusiveScan(input, output, initial, scan_op, aggregate);
        }

    //! Broadcast a value to logical warp.
    /*!
     * \param input Thread data to broadcast.
     * \param src_lane Index within logical warp to broadcast from.
     * \returns Broadcast value from \a src_lane
     *
     * For \a input <tt>{1,2,3,4}</tt>, broadcasting from \a src_lane 0 would return <tt>1</tt>.
     */
    DEVICE T Broadcast(T input, unsigned int src_lane)
        {
        // shuffle-based broadcast does not need temporary space, so we let the compiler optimize
        // this dummy variable out
        TempStorage tmp;
        return MyWarpScan(tmp).Broadcast(input, src_lane);
        }

    private:
#ifdef __HIP_PLATFORM_HCC__
    typedef hipcub::WarpScan<T, LOGICAL_WARP_THREADS, PTX_ARCH>
        MyWarpScan; //!< CUB shuffle-based scan
#else
    typedef cub::WarpScan<T, LOGICAL_WARP_THREADS, PTX_ARCH> MyWarpScan; //!< CUB shuffle-based scan
#endif
    typedef typename MyWarpScan::TempStorage
        TempStorage; //!< Nominal data type for CUB temporary storage

#ifdef __HIP_PLATFORM_HCC__
    static_assert(std::is_empty<TempStorage>::value, "WarpScan requires temp storage ");
#else
// we would like to make a similar guarantee with NVIDA CUB too, but TempStorage is not an empty
// type even if it internally uses WarpScanShfl
// static_assert(std::is_empty<TempStorage>::value, "WarpScan requires temp storage ");
#endif
    };

    } // end namespace detail
    } // end namespace hoomd

#undef DEVICE
