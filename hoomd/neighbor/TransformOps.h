// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Original license
// Copyright (c) 2018, Michael P. Howard.
// This file is released under the Modified BSD License.

// Maintainer: mphoward

#ifndef NEIGHBOR_TRANSFORM_OPS_H_
#define NEIGHBOR_TRANSFORM_OPS_H_

#ifdef NVCC
#define HOSTDEVICE __host__ __device__ __forceinline__
#else
#define HOSTDEVICE
#endif

namespace neighbor
{

//! Mapping operation for an index
/*!
 * A TransformOp is used to transform a primitive index into another cached value for traversal.
 * This may be useful for saving pointer indirection during traversal if, for example, the primitives used to
 * build the LBVH have already been sorted or correspond to a shifted range.
 *
 * The MapTransformOp is a reference implementation of a TransformOp. It takes a given primitive index and
 * returns the index mapped to a new value. Each TransformOp must supply an operator to perform this transformation.
 */
struct MapTransformOp
    {
    //! Constructor
    /*!
     * \param map_ A map of primitive indexes to their another value.
     */
    MapTransformOp(const unsigned int* map_)
        : map(map_)
        {}

    //! Mapping operator
    /*!
     * \param primitive The primitive index to transform.
     * \returns The mapped primitive index.
     */
    HOSTDEVICE unsigned int operator()(unsigned int primitive) const
        {
        return map[primitive];
        }

    const unsigned int* map;    //!< Map operation
    };


//! No transformation
/*!
 * The NullTransformOp simply returns the original primitive index without applying an transformation.
 */
struct NullTransformOp
    {
    //! Mapping operator
    /*!
     * \param primitive The primitive index to transform.
     * \returns The \a primitive without any transformation.
     */
    HOSTDEVICE unsigned int operator()(unsigned int primitive) const
        {
        return primitive;
        }
    };

} // end namespace neighbor

#undef HOSTDEVICE

#endif // NEIGHBOR_TRANSFORM_OPS_H_
