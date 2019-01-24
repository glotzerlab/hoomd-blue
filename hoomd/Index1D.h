// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __INDEX1D_H__
#define __INDEX1D_H__

/*! \file Index1D.h
    \brief Defines utility classes for 1D indexing of multi-dimensional arrays
    \details These are very low-level, high performance functions. No error checking is performed on their arguments,
    even in debug mode. They are provided mainly as a convenience. The col,row ordering is unorthodox for normal
    matrices, but is consistent with the tex2D x,y access pattern used in CUDA. The decision is to go with x,y for
    consistency.
*/

#include "HOOMDMath.h"

// need to declare these classes with __host__ __device__ qualifiers when building in nvcc
// HOSTDEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

//! Index a 2D array
/*! Row major mapping of 2D onto 1D
    \ingroup utils
*/
struct Index2D
    {
    public:
        //! Constructor
        /*! \param w Width of the square 2D array
        */
        HOSTDEVICE inline Index2D(unsigned int w=0) : m_w(w), m_h(w) {}

        //! Constructor
        /*! \param w Width of the rectangular 2D array
            \param h Height of the rectangular 2D array
        */
        HOSTDEVICE inline Index2D(unsigned int w, unsigned int h) : m_w(w), m_h(h) {}

        //! Calculate an index
        /*! \param i column index
            \param j row index
            \returns 1D array index corresponding to the 2D index (\a i, \a j) in row major order
        */
        HOSTDEVICE inline unsigned int operator()(unsigned int i, unsigned int j) const
            {
            return j*m_w + i;
            }

        //! Get the number of 1D elements stored
        /*! \returns Number of elements stored in the underlying 1D array
        */
        HOSTDEVICE inline unsigned int getNumElements() const
            {
            return m_w*m_h;
            }

        //! Get the width of the 2D array
        HOSTDEVICE inline unsigned int getW() const
            {
            return m_w;
            }

        //! Get the height of the 2D array
        HOSTDEVICE inline unsigned int getH() const
            {
            return m_h;
            }

        //! Get the inverse mapping 1D-index -> coordinate pair
        HOSTDEVICE inline uint2 getPair(const unsigned int idx) const
            {
            uint2 t;

            t.y = idx / m_w;
            t.x = idx % m_w;
            return t;
            }


    private:
        unsigned int m_w;   //!< Width of the 2D array
        unsigned int m_h;   //!< Height of the 2D array
    };

//! Index a 3D array
/*! Row major mapping of 3D onto 1D
    \ingroup utils
*/
struct Index3D
    {
    public:
        //! Constructor
        /*! \param w Width of the square 3D array
        */
        HOSTDEVICE inline Index3D(unsigned int w=0) : m_w(w), m_h(w), m_d(w) {}

        //! Constructor
        /*! \param w Width of the rectangular 3D array
            \param h Height of the rectangular 3D array
            \param d Depth of the rectangular 3D array
        */
        HOSTDEVICE inline Index3D(unsigned int w, unsigned int h, unsigned int d) : m_w(w), m_h(h), m_d(d) {}

        //! Calculate an index
        /*! \param i column index (along width)
            \param j row index (along height)
            \param k plane index (along depth)
            \returns 1D array index corresponding to the 2D index (\a i, \a j) in row major order
        */
        HOSTDEVICE inline unsigned int operator()(unsigned int i, unsigned int j, unsigned int k) const
            {
            return (k*m_h + j)*m_w + i;
            }

        //! Get the number of 1D elements stored
        /*! \returns Number of elements stored in the underlying 1D array
        */
        HOSTDEVICE inline unsigned int getNumElements() const
            {
            return m_w * m_h * m_d;
            }

        //! Get the width of the 3D array
        HOSTDEVICE inline unsigned int getW() const
            {
            return m_w;
            }

        //! Get the height of the 3D array
        HOSTDEVICE inline unsigned int getH() const
            {
            return m_h;
            }

        //! Get the depth of the 3D array
        HOSTDEVICE inline unsigned int getD() const
            {
            return m_d;
            }

        //! Get the inverse mapping 1D-index -> coordinate tuple
        HOSTDEVICE inline uint3 getTriple(const unsigned int idx) const
            {
            uint3 t;

            t.z = idx / (m_h*m_w);
            t.y = (idx % (m_h*m_w)) / m_w;
            t.x = idx - t.z * m_h *m_w - t.y * m_w;
            return t;
            }

    private:
        unsigned int m_w;   //!< Width of the 3D array
        unsigned int m_h;   //!< Height of the 3D array
        unsigned int m_d;   //!< Depth of the 3D array
    };

//! Index a 2D upper triangular array
/*! Row major mapping of a 2D upper triangular array onto 1D
    \ingroup utils
*/
struct Index2DUpperTriangular
    {
    public:
        //! Constructor
        /*! \param w Width of the 2D upper triangular array
        */
        HOSTDEVICE inline Index2DUpperTriangular(unsigned int w=0) : m_w(w)
            {
            m_term = 2*m_w - 1;
            }

        //! Calculate an index
        /*! \param i column index
            \param j row index
            \returns 1D array index corresponding to the 2D index (\a i, \a j) in row major order
            \note Formula adapted from: http://www.itl.nist.gov/div897/sqg/dads/HTML/upperTriangularMatrix.html
        */
        HOSTDEVICE inline unsigned int operator()(unsigned int i, unsigned int j) const
            {
            // swap if j > i
            if (j > i)
                {
                unsigned int tmp = i;
                i = j;
                j = tmp;
                }
            return j * (m_term - j) / 2 + i;
            }

        //! Get the number of 1D elements stored
        /*! \returns Number of elements stored in the underlying 1D array
        */
        HOSTDEVICE inline unsigned int getNumElements() const
            {
            return m_w*(m_w+1) / 2;
            }

    private:
        unsigned int m_w;     //!< Width of the 2D upper triangular array
        unsigned int m_term;  //!< Precomputed term of the equation for efficiency
    };

#undef HOSTDEVICE
#endif
