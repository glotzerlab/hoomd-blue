// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BoxDim.h
    \brief Defines the BoxDim class
*/

#ifndef __BOXDIM_H__
#define __BOXDIM_H__

#include "HOOMDMath.h"
#include "VectorMath.h"

#include <array>

// Don't include MPI when compiling with __HIPCC__
#if defined(ENABLE_MPI) && !defined(__HIPCC__)
#include "HOOMDMPI.h"
#endif

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#endif

namespace hoomd
    {
//! Stores box dimensions
/*! All particles in the ParticleData structure are inside of a box. This struct defines
    that box. For cubic boxes, inside is defined as x >= m_lo.x && x < m_hi.x, and similarly for y
   and z.

    For triclinic boxes, tilt factors xy, xz and yz are defined. In this case, m_lo and m_hi are the
   corners of the corresponding cubic box, for which the tilt factors would be zero.

    The conditions for a particle to be inside the triclinic box are

                              -m_L.z/2 <= z <= m_L.z/2
                       -m_L.y/2 + yz*z <= y <= m_L.y/2 + yz*z
        -m_L.x/2 + (xz-xy*yz)*z + xy*y <= x <= m_L.x/2 + (xz-xy*yz)*z + xy*y

    Boxes constructed via length default to periodic in all 3 directions. Any direction may be made
   non-periodic with setPeriodic(). Boxes constructed via lo and hi must be explicitly given
   periodic flags for each direction. The lo value \b must equal the negative for the high value for
   any direction that is set periodic. This is due to performance optimizations used in the minimum
   image convention. Callers that specify lo and hi directly must be aware of this fact. BoxDim does
   not check for erroneous input regarding \a lo, \a hi and the periodic flags. getPeriodic() can be
   used to query which directions are periodic.

    setL() can be used to update boxes where lo == -hi in all directions. setLoHi() can be used to
   update boxes where this is not the case.

    BoxDim comes with several analysis/computation methods to aid in working with vectors in boxes.
     - makeFraction() takes a vector in a box and computes a vector where all components are between
   0 and 1. 0,0,0 is lo and 1,1,1 is hi with a linear interpolation between.
     - minImage() takes a vector and wraps it back into the box following the minimum image
   convention, but only for those dimensions that are set periodic
     - wrap() wraps a vector back into the box and updates an image flag variable appropriately when
   particles cross box boundaries. It does this only for dimensions that are set periodic

    \note minImage() and wrap() only work for particles that have moved up to 1 box image out of the
   box.
*/
struct
#ifndef __HIPCC__
    __attribute__((visibility("default")))
#endif
    BoxDim
    {
    public:
    //! Constructs a useless box
    /*! \post All dimensions are 0.0
     */
    HOSTDEVICE explicit BoxDim()
        {
        m_lo = m_hi = m_Linv = m_L = make_scalar3(0, 0, 0);
        m_xz = m_xy = m_yz = Scalar(0.0);
        m_periodic = make_uchar3(1, 1, 1);
        }

    //! Constructs a box from -Len/2 to Len/2
    /*! \param Len Length of one side of the box
        \post Box ranges from \c -Len/2 to \c +Len/2 in all 3 dimensions
        \post periodic = (1,1,1)
    */
    HOSTDEVICE explicit BoxDim(Scalar Len)
        {
        setL(make_scalar3(Len, Len, Len));
        m_periodic = make_uchar3(1, 1, 1);
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
        m_periodic = make_uchar3(1, 1, 1);
        m_xz = m_xy = m_yz = Scalar(0.0);
        }

    //! Constructs a box from -L/2 to L/2 for each dimension
    /*! \param L box lengths
        \post periodic = (1,1,1)
    */
    HOSTDEVICE explicit BoxDim(Scalar3 L)
        {
        setL(L);
        m_periodic = make_uchar3(1, 1, 1);
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
        setL(make_scalar3(Len, Len, Len));
        setTiltFactors(xy, xz, yz);
        m_periodic = make_uchar3(1, 1, 1);
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

    /// Constructs a box from a std::array<Scalar, 6>
    /** @param array Box parameters
     */
    explicit BoxDim(const std::array<Scalar, 6>& array)
        {
        setL(make_scalar3(array[0], array[1], array[2]));
        setTiltFactors(array[3], array[4], array[5]);
        m_periodic = make_uchar3(1, 1, 1);
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
        \note It is invalid to set 1 for a periodic dimension where lo != -hi. This error is not
       checked for.
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
        m_hi = L / Scalar(2.0);
        m_lo = -m_hi;
        m_Linv = Scalar(1.0) / L;

        // avoid NaN when Lz == 0
        if (L.z == Scalar(0.0))
            {
            m_Linv.z = 0;
            }

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

        // avoid NaN when Lz == 0
        m_Linv = Scalar(1.0) / (m_hi - m_lo);
        if (m_hi.z == Scalar(0.0) && m_lo.z == Scalar(0.0))
            {
            m_Linv.z = 0;
            }
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
    HOSTDEVICE Scalar getTiltFactorXY() const
        {
        return m_xy;
        }

    //! Returns the xz tilt factor
    HOSTDEVICE Scalar getTiltFactorXZ() const
        {
        return m_xz;
        }

    //! Returns the yz tilt factor
    HOSTDEVICE Scalar getTiltFactorYZ() const
        {
        return m_yz;
        }

    //! Compute fractional coordinates, allowing for a ghost layer
    /*! \param v Vector to scale
        \param ghost_width Width of extra ghost padding layer to take into account (along reciprocal
       lattice directions) \return a vector with coordinates scaled to range between 0 and 1 (if
       inside the box + ghost layer). The returned vector \a f and the given vector \a v are related
       by: \a v = \a f * (L+2*ghost_width) + lo - ghost_width
    */
    HOSTDEVICE Scalar3 makeFraction(const Scalar3& v,
                                    const Scalar3& ghost_width = make_scalar3(0.0, 0.0, 0.0)) const
        {
        Scalar3 delta = v - m_lo;
        delta.x -= (m_xz - m_yz * m_xy) * v.z + m_xy * v.y;
        delta.y -= m_yz * v.z;
        Scalar3 ghost_frac = ghost_width / getNearestPlaneDistance();
        return (delta * m_Linv + ghost_frac) / (make_scalar3(1, 1, 1) + Scalar(2.0) * ghost_frac);
        }

    //! Make fraction using vec3s
    HOSTDEVICE vec3<Scalar> makeFraction(const vec3<Scalar>& v,
                                         const Scalar3& ghost_width
                                         = make_scalar3(0.0, 0.0, 0.0)) const
        {
        return vec3<Scalar>(makeFraction(vec_to_scalar3(v), ghost_width));
        }

    //! Convert fractional coordinates into real coordinates
    /*! \param f Fractional coordinates between 0 and 1 to scale
        \return A vector inside the box corresponding to f
     */
    HOSTDEVICE Scalar3 makeCoordinates(const Scalar3& f) const
        {
        Scalar3 v = m_lo + f * m_L;
        v.x += m_xy * v.y + m_xz * v.z;
        v.y += m_yz * v.z;
        return v;
        }

    //! makeCoordinates for vec3
    HOSTDEVICE vec3<Scalar> makeCoordinates(const vec3<Scalar>& f) const
        {
        return vec3<Scalar>(makeCoordinates(vec_to_scalar3(f)));
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

#ifdef __HIPCC__
        if (m_periodic.z)
            {
            Scalar img = slow::rint(w.z * m_Linv.z);
            w.z -= L.z * img;
            w.y -= L.z * m_yz * img;
            w.x -= L.z * m_xz * img;
            }

        if (m_periodic.y)
            {
            Scalar img = slow::rint(w.y * m_Linv.y);
            w.y -= L.y * img;
            w.x -= L.y * m_xy * img;
            }

        if (m_periodic.x)
            {
            w.x -= L.x * slow::rint(w.x * m_Linv.x);
            }
#else
        // on the cpu, branches are faster than calling rint
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
                int i = int(w.y * m_Linv.y + Scalar(0.5));
                w.y -= (Scalar)i * L.y;
                w.x -= (Scalar)i * L.y * m_xy;
                }
            else if (w.y < m_lo.y)
                {
                int i = int(-w.y * m_Linv.y + Scalar(0.5));
                w.y += (Scalar)i * L.y;
                w.x += (Scalar)i * L.y * m_xy;
                }
            }

        if (m_periodic.x)
            {
            if (w.x >= m_hi.x)
                {
                int i = int(w.x * m_Linv.x + Scalar(0.5));
                w.x -= (Scalar)i * L.x;
                }
            else if (w.x < m_lo.x)
                {
                int i = int(-w.x * m_Linv.x + Scalar(0.5));
                w.x += (Scalar)i * L.x;
                }
            }
#endif

        return w;
        }

    //! Minimum image using vec3s
    HOSTDEVICE vec3<Scalar> minImage(const vec3<Scalar>& v) const
        {
        return vec3<Scalar>(minImage(vec_to_scalar3(v)));
        }

    //! Wrap a vector back into the box
    /*! \param w Vector to wrap, updated to the minimum image obeying the periodic settings
        \param img Image of the vector, updated to reflect the new image
        \param flags Vector of flags to force wrapping along certain directions
        \post \a img and \a v are updated appropriately
        \note \a v must not extend more than 1 image beyond the box
    */
    HOSTDEVICE void wrap(Scalar3& w, int3& img, char3 flags = make_char3(0, 0, 0)) const
        {
        Scalar3 L = getL();

        // allow for a shifted box with periodic boundary conditions
        Scalar3 origin = (m_hi + m_lo) / Scalar(2.0);

        // tilt factors for nonperiodic boxes are always calculated w.r.t. to the global box with
        // origin (0,0,0)
        if (!m_periodic.y)
            {
            origin.y = Scalar(0.0);
            }
        if (!m_periodic.z)
            {
            origin.z = Scalar(0.0);
            }

        if (m_periodic.x)
            {
            Scalar tilt_x = (m_xz - m_xy * m_yz) * (w.z - origin.z) + m_xy * (w.y - origin.y);
            if (((w.x >= m_hi.x + tilt_x) && !flags.x) || flags.x == 1)
                {
                w.x -= L.x;
                img.x++;
                }
            else if (((w.x < m_lo.x + tilt_x) && !flags.x) || flags.x == -1)
                {
                w.x += L.x;
                img.x--;
                }
            }

        if (m_periodic.y)
            {
            Scalar tilt_y = m_yz * (w.z - origin.z);
            if (((w.y >= m_hi.y + tilt_y) && !flags.y) || flags.y == 1)
                {
                w.y -= L.y;
                w.x -= L.y * m_xy;
                img.y++;
                }
            else if (((w.y < m_lo.y + tilt_y) && !flags.y) || flags.y == -1)
                {
                w.y += L.y;
                w.x += L.y * m_xy;
                img.y--;
                }
            }

        if (m_periodic.z)
            {
            if (((w.z >= m_hi.z) && !flags.z) || flags.z == 1)
                {
                w.z -= L.z;
                w.y -= L.z * m_yz;
                w.x -= L.z * m_xz;
                img.z++;
                }
            else if (((w.z < m_lo.z) && !flags.z) || flags.z == -1)
                {
                w.z += L.z;
                w.y += L.z * m_yz;
                w.x += L.z * m_xz;
                img.z--;
                }
            }
        }

    //! Wrap a vec3
    HOSTDEVICE void wrap(vec3<Scalar>& w, int3& img, char3 flags = make_char3(0, 0, 0)) const
        {
        Scalar3 w_scalar = vec_to_scalar3(w);
        wrap(w_scalar, img, flags);
        w.x = w_scalar.x;
        w.y = w_scalar.y;
        w.z = w_scalar.z;
        }

    //! Wrap a vector back into the box
    /*! \param w Vector to wrap, updated to the minimum image obeying the periodic settings
        \param img Image of the vector, updated to reflect the new image
        \param flags Vector of flags to force wrapping along certain directions
        \post \a img and \a v are updated appropriately
        \note \a v must not extend more than 1 image beyond the box
        \note This is a special version that wraps a Scalar4 (the 4th element is left alone)
    */
    HOSTDEVICE void wrap(Scalar4& w, int3& img, char3 flags = make_char3(0, 0, 0)) const
        {
        Scalar3 v = make_scalar3(w.x, w.y, w.z);
        wrap(v, img, flags);
        w.x = v.x;
        w.y = v.y;
        w.z = v.z;
        }

    //! Get the periodic image a vector belongs to
    /*! \param v The vector to check
        \returns the integer coordinates of the periodic image
     */
    HOSTDEVICE int3 getImage(const Scalar3& v) const
        {
        Scalar3 f = makeFraction(v) - make_scalar3(0.5, 0.5, 0.5);
        int3 img;
        img.x = m_periodic.x ? ((int)((f.x >= Scalar(0.0)) ? f.x + Scalar(0.5) : f.x - Scalar(0.5)))
                             : 0;
        img.y = m_periodic.y ? ((int)((f.y >= Scalar(0.0)) ? f.y + Scalar(0.5) : f.y - Scalar(0.5)))
                             : 0;
        img.z = m_periodic.z ? ((int)((f.z >= Scalar(0.0)) ? f.z + Scalar(0.5) : f.z - Scalar(0.5)))
                             : 0;
        return img;
        }

    HOSTDEVICE int3 getImage(const vec3<Scalar>& v) const
        {
        return getImage(vec_to_scalar3(v));
        }

    //! Shift a vector by a multiple of the lattice vectors
    /*! \param v The vector to shift
        \param shift The displacement in lattice coordinates
     */
    HOSTDEVICE Scalar3 shift(const Scalar3& v, const int3& shift) const
        {
        Scalar3 r = v;
        r += Scalar(shift.x) * getLatticeVector(0);
        r += Scalar(shift.y) * getLatticeVector(1);
        r += Scalar(shift.z) * getLatticeVector(2);
        return r;
        }

    //! Shift a vec3
    HOSTDEVICE vec3<Scalar> shift(const vec3<Scalar>& v, const int3& _shift) const
        {
        return vec3<Scalar>(shift(vec_to_scalar3(v), _shift));
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
        dist.x = m_L.x
                 * fast::rsqrt(Scalar(1.0) + m_xy * m_xy
                               + (m_xy * m_yz - m_xz) * (m_xy * m_yz - m_xz));
        dist.y = m_L.y * fast::rsqrt(Scalar(1.0) + m_yz * m_yz);
        dist.z = m_L.z;

        // avoid NaN when Lz == 0
        if (m_L.z == Scalar(0.0))
            {
            dist.z = Scalar(1.0);
            }

        return dist;
        }

    //! Get the volume of the box
    /*! \returns the volume
     *  \param twod If true, return the area instead of the volume
     */
    HOSTDEVICE Scalar getVolume(bool twod = false) const
        {
        if (twod)
            return m_L.x * m_L.y;
        else
            return m_L.x * m_L.y * m_L.z;
        }

    /*! Get the lattice vector with index i

        \param i Index (0<=i<=2) of the lattice vector
        \returns the lattice vector with index i, or (0,0,0) if i is invalid
     */
    HOSTDEVICE Scalar3 getLatticeVector(unsigned int i) const
        {
        if (i == 0)
            {
            return make_scalar3(m_L.x, 0.0, 0.0);
            }
        else if (i == 1)
            {
            return make_scalar3(m_L.y * m_xy, m_L.y, 0.0);
            }
        else if (i == 2)
            {
            return make_scalar3(m_L.z * m_xz, m_L.z * m_yz, m_L.z);
            }

        return make_scalar3(0.0, 0.0, 0.0);
        }

    HOSTDEVICE bool operator==(const BoxDim& other) const
        {
        Scalar3 L1 = getL();
        Scalar3 L2 = other.getL();

        Scalar xy1 = getTiltFactorXY();
        Scalar xy2 = other.getTiltFactorXY();
        Scalar xz1 = getTiltFactorXZ();
        Scalar xz2 = other.getTiltFactorXZ();
        Scalar yz1 = getTiltFactorYZ();
        Scalar yz2 = other.getTiltFactorYZ();

        return L1 == L2 && xy1 == xy2 && xz1 == xz2 && yz1 == yz2;
        }

    HOSTDEVICE bool operator!=(const BoxDim& other) const
        {
        return !((*this) == other);
        }
#ifdef ENABLE_MPI
    //! Serialization method
    template<class Archive> void serialize(Archive& ar, const unsigned int version)
        {
        ar & m_lo.x;
        ar & m_lo.y;
        ar & m_lo.z;
        ar & m_hi.x;
        ar & m_hi.y;
        ar & m_hi.z;
        ar & m_L.x;
        ar & m_L.y;
        ar & m_L.z;
        ar & m_Linv.x;
        ar & m_Linv.y;
        ar & m_Linv.z;
        ar & m_xy;
        ar & m_xz;
        ar & m_yz;
        ar & m_periodic.x;
        ar & m_periodic.y;
        ar & m_periodic.z;
        }
#endif

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

    } // end namespace hoomd

// undefine HOSTDEVICE so we don't interfere with other headers
#undef HOSTDEVICE
#endif // __BOXDIM_H__
