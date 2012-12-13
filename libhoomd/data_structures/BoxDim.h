/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

/*! \file BoxDim.h
    \brief Defines the BoxDim class
*/

#ifndef __BOXDIM_H__
#define __BOXDIM_H__

#include "HOOMDMath.h"

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline
#endif
 
// call different optimized sqrt functions on the host / device
// SQRT is sqrtf when included in nvcc and sqrt when included into the host compiler
#ifdef NVCC
#define SQRT sqrtf
#else
#define SQRT sqrt
#endif

// RSQRT is rsqrtf when included in nvcc and 1.0 / sqrt(x) when included into the host compiler
#ifdef NVCC
#define RSQRT(x) rsqrtf( (x) )
#else
#define RSQRT(x) Scalar(1.0) / sqrt( (x) )
#endif

//! Stores box dimensions
/*! All particles in the ParticleData structure are inside of a box. This struct defines
    that box. For cubic boxes, inside is defined as x >= m_lo.x && x < m_hi.x, and similarly for y and z.
   
    For triclinic boxes, tilt factors xy, xz and yz are defined. In this case, m_lo and m_hi are the corners of the
    corresponding cubic box, for which the tilt factors would be zero.

    The conditions for a particle to be inside the triclinic box are

                              -m_L.z/2 <= z <= m_L.z/2
                       -m_L.y/2 + yz*z <= y <= m_L.y/2 + yz*z
        -m_L.x/2 + (xz-xy*yz)*z + xy*y <= x <= m_L.x/2 + (xz-xy*yz)*z + xy*y

    Boxes constructed via length default to periodic in all 3 directions. Any direction may be made non-periodic with
    setPeriodic(). Boxes constructed via lo and hi must be explicity given periodic flags for each direction.
    The lo value \b must equal the negative for the high value for any direction that is set periodic. This is due to
    performance optimizations used in the minimum image convention. Callers that specify lo and hi directly must be
    aware of this fact. BoxDim does not check for erroneous input regarding \a lo, \a hi and the periodic flags.
    getPeriodic() can be used to query which directions are periodic.
    
    setL() can be used to update boxes where lo == -hi in all directions. setLoHi() can be used to update boxes where
    this is not the case.
    
    BoxDim comes with several analysis/computaiton methods to aid in working with vectors in boxes.
     - makeFraction() takes a vector in a box and computes a vector where all components are between 0 and 1. 0,0,0 is 
       lo and 1,1,1 is hi with a linear interpolation between.
     - minImage() takes a vector and wraps it back into the box following the minimum image convention, but only for
       those dimensions that are set periodic
     - wrap() wraps a vector back into the box and updates an image flag variable appropriately when particles cross
       box boundaries. It does this only for dimensions that are set periodic
    
    \note minImage() and wrap() only work for particles that have moved up to 1 box image out of the box.
*/
class BoxDim
    {
    public:
        //! Constructs a useless box
        /*! \post All dimensions are 0.0
        */
        HOSTDEVICE explicit BoxDim()
            {
            m_lo = m_hi = m_Linv = m_L = make_scalar3(0, 0, 0);
            m_xz = m_xy = m_yz = Scalar(0.0);
            m_periodic = make_uchar3(1,1,1);
            }

        //! Constructs a box from -Len/2 to Len/2
        /*! \param Len Length of one side of the box
            \post Box ranges from \c -Len/2 to \c +Len/2 in all 3 dimensions
            \post periodic = (1,1,1)
        */
        HOSTDEVICE explicit BoxDim(Scalar Len)
            {
            setL(make_scalar3(Len, Len, Len));
            m_periodic = make_uchar3(1,1,1);
            m_xz = m_xy = m_yz = Scalar(0.0);
            }

        //! Constructs a box from -Len_x/2 to Len_x/2 for each dimension
        /*! \param Len_x Length of the x dimension of the box
            \param Len_y Length of the x dimension of the box
            \param Len_z Length of the x dimension of the box
            \post periodic = (1,1,1)
        */
        HOSTDEVICE explicit BoxDim(Scalar Len_x, Scalar Len_y, Scalar Len_z)
            {
            setL(make_scalar3(Len_x, Len_y, Len_z));
            m_periodic = make_uchar3(1,1,1);
            m_xz = m_xy = m_yz = Scalar(0.0);
            }

        //! Constructs a box from -L/2 to L/2 for each dimension
        /*! \param L box lengths
            \post periodic = (1,1,1)
        */
        HOSTDEVICE explicit BoxDim(Scalar3 L)
            {
            setL(L);
            m_periodic = make_uchar3(1,1,1);
            m_xz = m_xy = m_yz = Scalar(0.0);
            }

        //! Constructs a tilted box with edges of length len for each dimension
        /*! \param Len Box length
            \param xy Tilt factor of y-axis in xy plane
            \param xz Tilt factor of z-axis in xz plane
            \param yz Tilt factor of z-axis in yz plane
         */
        HOSTDEVICE explicit BoxDim(Scalar Len, Scalar xy, Scalar xz, Scalar yz)
            {
            setL(make_scalar3(Len,Len,Len));
            setTiltFactors(xy, xz, yz);
            m_periodic = make_uchar3(1,1,1);
            }

        //! Construct a box from specific lo and hi values
        /*! \param lo Lo coordinate in the box
            \param hi Hi coordinate in the box
            \param periodic Periodic flags
        */
        HOSTDEVICE explicit BoxDim(Scalar3 lo, Scalar3 hi, uchar3 periodic)
            {
            setLoHi(lo, hi);
            m_periodic = periodic;
            m_xz = m_xy = m_yz = Scalar(0.0);
            }

        //! Get the periodic flags
        /*! \return Periodic flags
        */
        HOSTDEVICE uchar3 getPeriodic() const
            {
            return m_periodic;
            }

        //! Set the periodic flags
        /*! \param periodic Flags to set
            \post Period flags are set to \a periodic
            \note It is invalid to set 1 for a periodic dimension where lo != -hi. This error is not checked for.
        */
        HOSTDEVICE void setPeriodic(uchar3 periodic)
            {
            m_periodic = periodic;
            }

        //! Get the length of the box in each direction
        /*! \returns The length of the box in each direction (hi - lo)
        */
        HOSTDEVICE Scalar3 getL() const
            {
            return m_L;
            }

        //! Update the box length
        /*! \param L new box length in each direction
        */
        HOSTDEVICE void setL(const Scalar3& L)
            {
            m_hi = L/Scalar(2.0);
            m_lo = -m_hi;
            m_Linv = Scalar(1.0)/L;
            m_L = L;
            }

        //! Get the lo coordinate of the box
        /*! \returns The lowest coordinate in the box
        */
        HOSTDEVICE Scalar3 getLo() const
            {
            return m_lo;
            }

        //! Get the hi coordinate of the box
        /*! \returns The highest coordinate in the box
        */
        HOSTDEVICE Scalar3 getHi() const
            {
            return m_hi;
            }

        //! Update the box lo and hi values
        /*! \param lo Lo coordinate in the box
            \param hi Hi coordinate in the box
        */
        HOSTDEVICE void setLoHi(const Scalar3& lo, const Scalar3& hi)
            {
            m_hi = hi;
            m_lo = lo;
            m_Linv = Scalar(1.0)/(m_hi - m_lo);
            m_L = m_hi - m_lo;
            }

        //! Update the box tilt factors
        /*! \param xy Tilt of y axis in x-y plane
            \param xz Tilt of z axis in x-z plane
            \param yz Tilt of z axis in x-y plane
         */
        HOSTDEVICE void setTiltFactors(const Scalar xy, const Scalar xz, const Scalar yz)
            {
            m_xy = xy;
            m_xz = xz;
            m_yz = yz;
            }

        //! Returns the xy tilt factor
        HOSTDEVICE Scalar getTiltFactorXY()
            {
            return m_xy;
            }

        //! Returns the xz tilt factor
        HOSTDEVICE Scalar getTiltFactorXZ()
            {
            return m_xz;
            }

        //! Returns the yz tilt factor
        HOSTDEVICE Scalar getTiltFactorYZ()
            {
            return m_yz;
            }

        //! Compute fractional coordinates, allowing for a ghost layer
        /*! \param v Vector to scale
            \param ghost_width Width of extra ghost padding layer to take into account
            \return a vector with coordinates scaled to range between 0 and 1 (if inside the box + ghost layer).
            The returned vector \a f and the given vector \a v are related by:
            \a v = \a f * (L+2*ghost_width) + lo - ghost_width
        */
        HOSTDEVICE Scalar3 makeFraction(const Scalar3& v, const Scalar3& ghost_width=make_scalar3(0.0,0.0,0.0)) const
            {
            Scalar3 delta = v - m_lo;
            delta.x -= (m_xz-m_yz*m_xy)*v.z+m_xy*v.y;
            delta.y -= m_yz * v.z;
            return (delta + ghost_width)/ (m_L + Scalar(2.0)*ghost_width);
            }

        //! Convert fractional coordinates into real coordinates
        /*! \param f Fractional coordinates between 0 and 1 to scale
            \return A vector inside the box corresponding to f
         */
        HOSTDEVICE Scalar3 makeCoordinates(const Scalar3 &f) const
            {
            Scalar3 v = m_lo + f*m_L;
            v.x += m_xy*v.y+m_xz*v.z;
            v.y += m_yz*v.z;
            return v;
            }

        //! Compute minimum image
        /*! \param v Vector to compute
            \return a vector that is the minimum image vector of \a v, obeying the periodic settings
            \note \a v must not extend more than 1 image beyond the box
        */
        HOSTDEVICE Scalar3 minImage(const Scalar3& v) const
            {
            Scalar3 w = v;
            Scalar3 L = getL();

            #ifdef NVCC
            if (m_periodic.z)
                {
                Scalar img = rintf(w.z * m_Linv.z);
                w.z -= L.z * img;
                w.y -= L.z * m_yz * img;
                w.x -= L.z * m_xz * img;
                }

            if (m_periodic.y)
                {
                Scalar img = rintf(w.y * m_Linv.y);
                w.y -= L.y * img;
                w.x -= L.y * m_xy * img;
                }

            if (m_periodic.x)
                {
                w.x -= L.x * rintf(w.x * m_Linv.x);
                }
            #else
            // on the cpu, branches are faster than calling rintf
            if (m_periodic.z)
                {
                if (w.z >= m_hi.z)
                    {
                    w.z -= L.z;
                    w.y -= L.z * m_yz;
                    w.x -= L.z * m_xz;
                    }
                else if (w.z < m_lo.z)
                    {
                    w.z += L.z;
                    w.y += L.z * m_yz;
                    w.x += L.z * m_xz;
                    }
                }
 
            if (m_periodic.y)
                {
                if (w.y >= m_hi.y)
                    {
                    w.y -= L.y;
                    w.x -= L.y * m_xy;
                    }
                else if (w.y < m_lo.y)
                    {
                    w.y += L.y;
                    w.x += L.y * m_xy;
                    }
                }

            if (m_periodic.x)
                {
                if (w.x >= m_hi.x)
                    w.x -= L.x;
                else if (w.x < m_lo.x)
                    w.x += L.x;
                }

            #endif
            return w;
            }

        //! Wrap a vector back into the box
        /*! \param w Vector to wrap, updated to the minimum image obeying the periodic settings
            \param img Image of the vector, updated to reflect the new image
            \post \a img and \a v are updated appropriately
            \note \a v must not extend more than 1 image beyond the box
        */
        HOSTDEVICE void wrap(Scalar3& w, int3& img) const
            {
            Scalar3 L = getL();

            if (m_periodic.x)
                {
                Scalar tilt_x = (m_xz - m_xy*m_yz) * w.z + m_xy * w.y;
                if (w.x >= m_hi.x + tilt_x)
                    {
                    w.x -= L.x;
                    img.x++;
                    }
                else if (w.x < m_lo.x + tilt_x)
                    {
                    w.x += L.x;
                    img.x--;
                    }
                }

            if (m_periodic.y)
                {
                Scalar tilt_y = m_yz * w.z;
                if (w.y >= m_hi.y + tilt_y)
                    {
                    w.y -= L.y;
                    w.x -= L.y * m_xy;
                    img.y++;
                    }
                else if (w.y < m_lo.y + tilt_y)
                    {
                    w.y += L.y;
                    w.x += L.y * m_xy;
                    img.y--;
                    }
                }

            if (m_periodic.z)
                {
                if (w.z >= m_hi.z)
                    {
                    w.z -= L.z;
                    w.y -= L.z * m_yz;
                    w.x -= L.z * m_xz;
                    img.z++;
                    }
                else if (w.z < m_lo.z)
                    {
                    w.z += L.z;
                    w.y += L.z * m_yz;
                    w.x += L.z * m_xz;
                    img.z--;
                    }
                }
           }

        //! Wrap a vector back into the box
        /*! \param w Vector to wrap, updated to the minimum image obeying the periodic settings
            \param img Image of the vector, updated to reflect the new image
            \post \a img and \a v are updated appropriately
            \note \a v must not extend more than 1 image beyond the box
            \note This is a special version that wraps a Scalar4 (the 4th element is left alone)
        */
        HOSTDEVICE void wrap(Scalar4& w, int3& img) const
            {
            Scalar3 v = make_scalar3(w.x, w.y, w.z);
            wrap(v, img);
            w.x = v.x;
            w.y = v.y;
            w.z = v.z;
            }

        //! Get the periodic image a vector belongs to
        /*! \param v The vector to check
            \returns the integer coordinates of the periodic image
         */
        HOSTDEVICE int3 getImage(Scalar3 &v) const
            {
            Scalar3 f = makeFraction(v) - make_scalar3(0.5,0.5,0.5);
            int3 img;
            img.x = (int)((f.x >= Scalar(0.0)) ? f.x + Scalar(0.5) : f.x - Scalar(0.5));
            img.y = (int)((f.y >= Scalar(0.0)) ? f.y + Scalar(0.5) : f.y - Scalar(0.5));
            img.z = (int)((f.z >= Scalar(0.0)) ? f.z + Scalar(0.5) : f.z - Scalar(0.5));
            return img;
            }


        //! Shift a vector by a multiple of the lattice vectors
        /*! \param v The vector to shift
            \param shift The displacement in lattice coordinates

            \note This method only works on boxes for which hi=-lo in all directions 
         */
        HOSTDEVICE void shift(Scalar3& v, int3& shift) const
            {
            v += makeCoordinates(make_scalar3(0.5,0.5,0.5)+make_scalar3(shift.x,shift.y,shift.z));
            }

        //! Check if the displacement is out of bounds
        /*! \param dx The displacement vector
            \returns True if the displacement exceeds the box length in a direction where periodic
                     boundary conditions are not applied
         */
        HOSTDEVICE bool checkOutOfBounds(Scalar3& dx) const
            {
            Scalar3 del;
            del.x = dx.x - (m_xz - m_xy*m_yz) * dx.z - m_xy * dx.y;
            del.y = dx.y -  m_yz * dx.z;
            del.z = dx.z;

            if (!m_periodic.x && del.x*del.x >= m_L.x*m_L.x) return true;
            if (!m_periodic.y && del.y*del.y >= m_L.y*m_L.y) return true;
            if (!m_periodic.z && del.z*del.z >= m_L.z*m_L.z) return true;

            return false;
            }

        //! Get the shortest distance between opposite boundary planes of the box
        /*! The distance between two planes of the lattice is 2 Pi/|b_i|, where
         *   b_1 is the reciprocal lattice vector of the Bravais lattice normal to
         *   the lattice vectors a_2 and a_3 etc.
         *
         * \return A Scalar3 containing the distance between the a_2-a_3, a_3-a_1 and
         *         a_1-a_2 planes for the triclinic lattice         
         */
        HOSTDEVICE Scalar3 getNearestPlaneDistance() const
            {
            Scalar3 dist;
            dist.x = m_L.x*RSQRT(Scalar(1.0) + m_xy*m_xy + (m_xy*m_yz - m_xz)*(m_xy*m_yz - m_xz));
            dist.y = m_L.y*RSQRT(Scalar(1.0) + m_yz*m_yz);
            dist.z = m_L.z;

            return dist;
            }

        //! Get the volume of the box
        /*! \returns the volume
         */
        HOSTDEVICE Scalar getVolume() const
            {
            return m_L.x*m_L.y*m_L.z;
            }

        /*! Get the lattice vector with index i
         
            \param i Index (0<=i<=2) of the lattice vector
            \returns the lattice vector with index i, or (0,0,0) if i is invalid
         */
        HOSTDEVICE Scalar3 getLatticeVector(unsigned int i) const
            {
            if (i == 0)
                {
                return make_scalar3(m_L.x,0.0,0.0);
                }
            else if (i == 1)
                {
                return make_scalar3(m_L.y*m_xy, m_L.y, 0.0);
                }
            else if (i == 2)
                {
                return make_scalar3(m_L.z*m_xz, m_L.z*m_yz, m_L.z);
                }

            return make_scalar3(0.0,0.0,0.0);
            }

    private:
        Scalar3 m_lo;      //!< Minimum coords in the box
        Scalar3 m_hi;      //!< Maximum coords in the box
        Scalar3 m_L;       //!< L precomputed (used to avoid subtractions in boundary conditions)
        Scalar3 m_Linv;    //!< 1/L precomputed (used to avoid divisions in boundary conditions)
        Scalar m_xy;       //!< xy tilt factor
        Scalar m_xz;       //!< xz tilt factor
        Scalar m_yz;       //!< yz tilt factor
        uchar3 m_periodic; //!< 0/1 in each direction to tell if the box is periodic in that direction
    };


// undefine HOSTDEVICE so we don't interfere with other headers
#undef HOSTDEVICE
#undef RSQRT
#undef SQRT
#endif // __BOXDIM_H__
