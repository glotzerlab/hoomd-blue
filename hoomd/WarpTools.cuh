// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file WarpTools.cuh
 * \brief Wrappers around CUB primitives for warp-level parallel primitives.
 */

#ifndef HOOMD_WARP_TOOLS_CUH_
#define HOOMD_WARP_TOOLS_CUH_

#if __CUDACC_VER_MAJOR__ >= 11
#include <cub/cub.cuh>
#else
#include "hoomd/extern/cub/cub/cub.cuh"
#endif

#define DEVICE __device__ __forceinline__

namespace hoomd
{
namespace detail
{

//! Computes warp-level reduction using shuffle instructions
/*!
 * Reduction operations are performed at the warp or sub-warp level using shuffle instructions. The sub-warp is defined as
 * a consecutive group of threads that is (1) smaller than the hardware warp size (32 threads) and (2) a power of 2.
 * For additional details about any operator, refer to the CUB documentation.
 *
 * This class is a thin wrapper around cub::WarpReduceShfl. The CUB scan classes nominally request "temporary" memory,
 * which is shared memory for non-shuffle scans. However, the shuffle-based scan does not use any shared memory,
 * and so this temporary variable is put unused into a register. The compiler can then optimize this out.
 * Care must be taken to monitor the CUB implementation in future to ensure the temporary memory is never used.
 *
 * \tparam T data type to scan
 * \tparam LOGICAL_WARP_THREADS number of threads in a "logical" warp, must be a multiple of 2.
 * \tparam PTX_ARCH PTX architecture to build for, must be at least 300 (Kepler).
 */
template<typename T, int LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS, int PTX_ARCH = CUB_PTX_ARCH>
class WarpReduce
    {
    public:
        DEVICE WarpReduce()
            {
            static_assert(PTX_ARCH >= 300, "PTX architecture must be >= 300");
            static_assert(LOGICAL_WARP_THREADS <= CUB_PTX_WARP_THREADS, "Logical warp size cannot exceed hardware warp size");
            static_assert(LOGICAL_WARP_THREADS && !(LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS-1)), "Logical warp size must be a power of 2");
            }

        //! Sum reduction.
        /*!
         * \param input Thread data to sum.
         * \returns output The result of the sum reduction in thread 0 of the (sub-)warp.
         *
         * The sum reduction for a 4-thread sub-warp with \a input <tt>{1,2,3,4}</tt> gives \a output <tt>10</tt> in thread 0.
         * The result in the other threads is undefined.
         */
        DEVICE T Sum(T input)
            {
            return Reduce(input, HOOMD_CUB::Sum());
            }

        //! Sum reduction over valid items.
        /*!
         * \param input Thread data to sum.
         * \param valid_items Total number of valid items in the (sub)-warp.
         * \returns output The result of the sum reduction in thread 0 of the (sub-)warp.
         *
         * The number of valid items may be smaller than the (sub-)warp. For example, if \a valid items is 3, then
         * the sum reduction for a 4-thread sub-warp with \a input <tt>{1,2,3,4}</tt> gives \a output <tt>6</tt> in thread 0.
         * The result in the other threads is undefined.
         */
        DEVICE T Sum(T input, int valid_items)
            {
            return Reduce(input, HOOMD_CUB::Sum(), valid_items);
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
        template<typename ReduceOpT>
        DEVICE T Reduce(T input, ReduceOpT reduce_op)
            {
            // shuffle-based reduce does not need temporary space, so we let the compiler optimize this dummy variable out
            TempStorage tmp;
            return WarpReduceShfl(tmp).template Reduce<true>(input, LOGICAL_WARP_THREADS, reduce_op);
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
        template<typename ReduceOpT>
        DEVICE T Reduce(T input, ReduceOpT reduce_op, int valid_items)
            {
            // shuffle-based reduce does not need temporary space, so we let the compiler optimize this dummy variable out
            TempStorage tmp;
            return WarpReduceShfl(tmp).template Reduce<false>(input, valid_items, reduce_op);
            }

    private:
        typedef HOOMD_CUB::WarpReduceShfl<T,LOGICAL_WARP_THREADS,PTX_ARCH> WarpReduceShfl;    //!< CUB shuffle-based reduce
        typedef typename WarpReduceShfl::TempStorage TempStorage;                       //!< Nominal data type for CUB temporary storage
    };

//! Computes warp-level scan (prefix sum) using shuffle instructions
/*!
 * Scan operations are performed at the warp or sub-warp level using shuffle instructions. The sub-warp is defined as
 * a consecutive group of threads that is (1) smaller than the hardware warp size (32 threads) and (2) a power of 2.
 * For additional details about any operator, refer to the CUB documentation.
 *
 * This class is a thin wrapper around cub::WarpScanShfl. The CUB scan classes nominally request "temporary" memory,
 * which is shared memory for non-shuffle scans. However, the shuffle-based scan does not use any shared memory,
 * and so this temporary variable is put unused into a register. The compiler can then optimize this out.
 * Care must be taken to monitor the CUB implementation in future to ensure the temporary memory is never used.
 *
 * \tparam T data type to scan
 * \tparam LOGICAL_WARP_THREADS number of threads in a "logical" warp, must be a multiple of 2.
 * \tparam PTX_ARCH PTX architecture to build for, must be at least 300 (Kepler).
 */
template<typename T, int LOGICAL_WARP_THREADS = CUB_PTX_WARP_THREADS, int PTX_ARCH = CUB_PTX_ARCH>
class WarpScan
    {
    public:
        DEVICE WarpScan()
            {
            static_assert(PTX_ARCH >= 300, "PTX architecture must be >= 300");
            static_assert(LOGICAL_WARP_THREADS <= CUB_PTX_WARP_THREADS, "Logical warp size cannot exceed hardware warp size");
            static_assert(LOGICAL_WARP_THREADS && !(LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS-1)), "Logical warp size must be a power of 2");
            }

        //! Inclusive sum for each thread in logical warp.
        /*!
         * \param input Thread data to sum.
         * \param output Result of scan for this thread.
         *
         * The inclusive sum for a 4-thread sub-warp with \a input <tt>{1,2,3,4}</tt> gives \a output <tt>{1,3,6,10}</tt>.
         */
        DEVICE void InclusiveSum(T input, T& output)
            {
            InclusiveScan(input, output, HOOMD_CUB::Sum());
            }

        //! Inclusive sum for each thread in logical warp, plus accumulation for all.
        /*!
         * \param input Thread data to sum.
         * \param output Result of scan for this thread.
         * \param aggregate Total sum of all threads.
         *
         * The inclusive sum for a 4-thread sub-warp with \a input <tt>{1,2,3,4}</tt> gives \a output <tt>{1,3,6,10}</tt>,
         * and the \a aggregate is 10 for all threads in the sub-warp.
         */
        DEVICE void InclusiveSum(T input, T& output, T& aggregate)
            {
            InclusiveScan(input, output, HOOMD_CUB::Sum(), aggregate);
            }

        //! Inclusive scan with a custom scan operator.
        /*!
         * \param input Thread data to sum.
         * \param output Result of scan for this thread.
         * \param scan_op Binary scan operator.
         *
         * This operator is equivalent to InclusiveSum if \a scan_op were cub::Sum().
         *
         * \tparam ScanOpT <b>inferred</b> Binary scan operator type.
         */
        template<class ScanOpT>
        DEVICE void InclusiveScan(T input, T& output, ScanOpT scan_op)
            {
            // shuffle-based scan does not need temporary space, so we let the compiler optimize this dummy variable out
            TempStorage tmp;
            WarpScanShfl(tmp).InclusiveScan(input, output, scan_op);
            }

        //! Inclusive scan with a custom scan operator, plus accumulation for all.
        /*!
         * \param input Thread data to sum.
         * \param output Result of scan for this thread.
         * \param scan_op Binary scan operator.
         * \param aggregate Total scan of all threads.
         *
         * This operator is equivalent to InclusiveSum if \a scan_op were cub::Sum().
         *
         * \tparam ScanOpT <b>inferred</b> Binary scan operator type.
         */
        template<class ScanOpT>
        DEVICE void InclusiveScan(T input, T& output, ScanOpT scan_op, T& aggregate)
            {
            // shuffle-based scan does not need temporary space, so we let the compiler optimize this dummy variable out
            TempStorage tmp;
            WarpScanShfl(tmp).InclusiveScan(input, output, scan_op, aggregate);
            }

        //! Exclusive sum for each thread in logical warp.
        /*!
         * \param input Thread data to sum.
         * \param output Result of scan for this thread.
         *
         * The inclusive sum for a 4-thread sub-warp with \a input <tt>{1,2,3,4}</tt> gives \a output <tt>{0,1,3,6}</tt>.
         * (The first thread is initialized from a value of zero.)
         */
        DEVICE void ExclusiveSum(T input, T& output)
            {
            T initial = 0;
            ExclusiveScan(input, output, initial, HOOMD_CUB::Sum());
            }

        //! Exclusive sum for each thread in logical warp, plus accumulation for all.
        /*!
         * \param input Thread data to sum.
         * \param output Result of scan for this thread.
         * \param aggregate Total sum of all threads.
         *
         * The inclusive sum for a 4-thread sub-warp with \a input <tt>{1,2,3,4}</tt> gives \a output <tt>{0,1,3,6}</tt>,
         * and the \a aggregate is 10 for all threads in the sub-warp.
         */
        DEVICE void ExclusiveSum(T input, T& output, T& aggregate)
            {
            T initial = 0;
            ExclusiveScan(input, output, initial, HOOMD_CUB::Sum(), aggregate);
            }

        //! Exclusive scan with a custom scan operator.
        /*!
         * \param input Thread data to sum.
         * \param output Result of scan for this thread.
         * \param scan_op Binary scan operator.
         *
         * This operator is equivalent to ExclusiveSum if \a scan_op were cub::Sum().
         *
         * \tparam ScanOpT <b>inferred</b> Binary scan operator type.
         */
        template<class ScanOpT>
        DEVICE void ExclusiveScan(T input, T& output, ScanOpT scan_op)
            {
            // shuffle-based scan does not need temporary space, so we let the compiler optimize this dummy variable out
            TempStorage tmp;
            WarpScanShfl scan(tmp);
            // first compute inclusive scan, then update to make exclusive
            T inclusive;
            scan.InclusiveScan(input, inclusive, scan_op);
            scan.Update(input, inclusive, output, scan_op, HOOMD_CUB::Int2Type<IS_INTEGER>());
            }

        //! Exclusive scan with a custom scan operator and initial value.
        /*!
         * \param input Thread data to sum.
         * \param output Result of scan for this thread.
         * \param initial Initial value for exclusive sum within logical warp.
         * \param scan_op Binary scan operator.
         *
         * This operator is equivalent to ExclusiveSum if \a scan_op were cub::Sum() and \a initial were zero.
         *
         * \tparam ScanOpT <b>inferred</b> Binary scan operator type.
         */
        template<class ScanOpT>
        DEVICE void ExclusiveScan(T input, T& output, T initial, ScanOpT scan_op)
            {
            // shuffle-based scan does not need temporary space, so we let the compiler optimize this dummy variable out
            TempStorage tmp;
            WarpScanShfl scan(tmp);
            // first compute inclusive scan, then update to make exclusive
            T inclusive;
            scan.InclusiveScan(input, inclusive, scan_op);
            scan.Update(input, inclusive, output, scan_op, initial, HOOMD_CUB::Int2Type<IS_INTEGER>());
            }

        //! Exclusive scan with a custom scan operator, plus accumulation for all.
        /*!
         * \param input Thread data to sum.
         * \param output Result of scan for this thread.
         * \param scan_op Binary scan operator.
         * \param aggregate Total scan of all threads.
         *
         * This operator is equivalent to ExclusiveSum if \a scan_op were cub::Sum().
         *
         * \tparam ScanOpT <b>inferred</b> Binary scan operator type.
         */
        template<class ScanOpT>
        DEVICE void ExclusiveScan(T input, T& output, ScanOpT scan_op, T& aggregate)
            {
            // shuffle-based scan does not need temporary space, so we let the compiler optimize this dummy variable out
            TempStorage tmp;
            WarpScanShfl scan(tmp);
            // first compute inclusive scan, then update to make exclusive
            T inclusive;
            scan.InclusiveScan(input, inclusive, scan_op);
            scan.Update(input, inclusive, output, aggregate, scan_op, HOOMD_CUB::Int2Type<IS_INTEGER>());
            }

        //! Exclusive scan with a custom scan operator and initial value, plus accumulation for all.
        /*!
         * \param input Thread data to sum.
         * \param output Result of scan for this thread.
         * \param initial Initial value for exclusive sum within logical warp.
         * \param scan_op Binary scan operator.
         * \param aggregate Total scan of all threads.
         *
         * This operator is equivalent to ExclusiveSum if \a scan_op were cub::Sum() and \a initial were zero.
         *
         * \tparam ScanOpT <b>inferred</b> Binary scan operator type.
         */
        template<class ScanOpT>
        DEVICE void ExclusiveScan(T input, T& output, T initial, ScanOpT scan_op, T& aggregate)
            {
            // shuffle-based scan does not need temporary space, so we let the compiler optimize this dummy variable out
            TempStorage tmp;
            WarpScanShfl scan(tmp);
            // first compute inclusive scan, then update to make exclusive
            T inclusive;
            scan.InclusiveScan(input, inclusive, scan_op);
            scan.Update(input, inclusive, output, aggregate, scan_op, initial, HOOMD_CUB::Int2Type<IS_INTEGER>());
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
            // shuffle-based broadcast does not need temporary space, so we let the compiler optimize this dummy variable out
            TempStorage tmp;
            return WarpScanShfl(tmp).Broadcast(input, src_lane);
            }

    private:
        typedef HOOMD_CUB::WarpScanShfl<T,LOGICAL_WARP_THREADS,PTX_ARCH> WarpScanShfl;    //!< CUB shuffle-based scan
        typedef typename WarpScanShfl::TempStorage TempStorage;                     //!< Nominal data type for CUB temporary storage

        enum
            {
            IS_INTEGER = ((HOOMD_CUB::Traits<T>::CATEGORY == HOOMD_CUB::SIGNED_INTEGER) || (HOOMD_CUB::Traits<T>::CATEGORY == HOOMD_CUB::UNSIGNED_INTEGER))
            };
    };

} // end namespace detail
} // end namespace hoomd

#undef DEVICE

#endif // HOOMD_WARP_TOOLS_CUH_
