// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file SinusoidalExpansionConstrictionGeometry.h
 * \brief Definition of the MPCD symmetric cosine channel geometry
 */

#ifndef MPCD_COSINE_EXPANSION_CONTRACTION_GEOMETRY_H_
#define MPCD_COSINE_EXPANSION_CONTRACTION_GEOMETRY_H_

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#include <string>
#endif // __HIPCC__

namespace hoomd
    {
namespace mpcd
    {

//! Sinusoidal expansion constriction channel geometry
/*!
 * This class defines a channel with a series of expansions and constrictions. Symmetric cosines
 * given by the equations +/-(A cos(x*2*pi*p/Lx) + A + contraction_separation*0.5) are used for the
 * walls. A = expansion_separation-contraction_separation is the amplitude and p is the period of
 * the wall cosine. expansion_separation is the height of the channel at its widest point,
 * contraction_separation is the half height of the channel at its narrowest point. The cosine wall
 * wavelength/frenquency needs to be consumable with the periodic boundary conditions in x,
 * therefore the period p is specified and the wavelength 2*pi*p/Lx is calculated.
 *
 * Below is an example how a cosine channel looks like in a 30x30x30 box with
 * expansion_separation=20, contraction_separation=2, and p=1. The wall cosine period p determines
 * how many repetitions of the geometry are in the simulation cell and there will be p wide
 * sections, centered at the origin of the simulation box.
 *
 *
 * 15 +-------------------------------------------------+
 *     |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 *     |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 *     |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 *  10 |XXXXXXXXXXXXXXXXXXX===========XXXXXXXXXXXXXXXXXXX|
 *     |XXXXXXXXXXXXXXX====           ====XXXXXXXXXXXXXXX|
 *     |XXXXXXXXXXXXX===                 ===XXXXXXXXXXXXX|
 *   5 |XXXXXXXXXXX==                       ==XXXXXXXXXXX|
 *     |XXXXXXXX===                           ===XXXXXXXX|
 *     |XXXXX====                               ====XXXXX|
 *     |=====                                       =====|
 * z 0 |                                                 |
 *     |=====                                       =====|
 *     |XXXXX====                               ====XXXXX|
 *     |XXXXXXXX===                           ===XXXXXXXX|
 *  -5 |XXXXXXXXXXX==                       ==XXXXXXXXXXX|
 *     |XXXXXXXXXXXXX===                 ===XXXXXXXXXXXXX|
 *     |XXXXXXXXXXXXXXX====           ====XXXXXXXXXXXXXXX|
 * -10 |XXXXXXXXXXXXXXXXXXX===========XXXXXXXXXXXXXXXXXXX|
 *     |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 *     |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 *     |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 * -15 +-------------------------------------------------+
 *    -15     -10      -5       0       5        10      15
 *                              x
 *
 * The wall boundary conditions can optionally be changed to slip conditions.
 *
 */
class __attribute__((visibility("default"))) CosineExpansionContractionGeometry
    {
    public:
    //! Constructor
    /*!
     * \param expansion_separation Channel width at narrowest point
       \param contraction_separation Channel width at widest point
       \param wavenumber Wavenumber of the cosine
     * \param no_slip Boundary condition at the wall (slip or no-slip)
     */
    HOSTDEVICE CosineExpansionContractionGeometry(Scalar expansion_separation,
                                                  Scalar contraction_separation,
                                                  Scalar wavenumber,
                                                  bool no_slip)
        : m_H_wide(Scalar(0.5) * expansion_separation),
          m_H_narrow(Scalar(0.5) * contraction_separation), m_wavenumber(wavenumber),
          m_no_slip(no_slip)
        {
        }

    //! Detect collision between the particle and the boundary
    /*!
     * \param pos Proposed particle position
     * \param vel Proposed particle velocity
     * \param dt Integration time remaining
     *
     * \returns True if a collision occurred, and false otherwise
     *
     * \post The particle position \a pos is moved to the point of reflection, the velocity \a vel
     * is updated according to the appropriate bounce back rule, and the integration time \a dt is
     * decreased to the amount of time remaining.
     */
    HOSTDEVICE bool detectCollision(Scalar3& pos, Scalar3& vel, Scalar& dt) const
        {
        /*
         * Detect if particle has left the box. The sign used
         * in calculations is +1 if the particle is out-of-bounds at the top wall, -1 if the
         * particle is out-of-bounds at the bottom wall, and 0 otherwise.
         *
         * We intentionally use > / < rather than >= / <= to make sure that spurious collisions do
         * not get detected when a particle is reset to the boundary location. A particle landing
         * exactly on the boundary from the bulk can be immediately reflected on the next streaming
         * step, and so the motion is essentially equivalent up to an epsilon of difference in the
         * channel width.
         */
        Scalar A = 0.5 * (m_H_wide - m_H_narrow);
        Scalar a = A * fast::cos(pos.x * m_wavenumber) + A + m_H_narrow;
        const signed char sign = (pos.y > a) - (pos.y < -a);

        // exit immediately if no collision is found
        if (sign == 0)
            {
            dt = Scalar(0);
            return false;
            }

        /* Calculate position (x0,y0,z0) of collision with wall:
         *  Because there is no analythical solution for equations like f(x) = cos(x)-x = 0, we use
         * Newtons's method or Bisection (if Newton fails) to nummerically estimate the x positon of
         * the intersection first. It is convinient to use the halfway point between the last
         * particle position outside the wall (at time t-dt) and the current position inside the
         * wall (at time t) as initial guess for the intersection.
         *
         *  We limit the number of iterations (max_iteration) and the desired presicion
         * (target_precision) for performance reasons. These values have been tested in python code
         * seperately and gave satisfactory results.
         *
         */
        const unsigned int max_iteration = 6;
        const Scalar target_precision = 1e-5;

        Scalar x0 = pos.x - 0.5 * dt * vel.x;
        Scalar y0;
        Scalar z0;
        Scalar n, n2;
        Scalar s, c;
        Scalar delta;

        // excatly horizontal z-collision, has a solution:
        if (vel.y == 0)
            {
            x0 = 1 / m_wavenumber * fast::acos((pos.y - A - m_H_narrow) / sign * A);
            y0 = pos.y;
            z0 = -(pos.x - dt * vel.x - x0) * vel.z / vel.x + (pos.z - dt * vel.z);
            }
        /* chatch the case where a particle collides exactly vertically (v_x=0 -> old x pos = new x
         * pos) In this case in Newton's method one would get: y0 = -(0)*0/0 + (y-dt*v_y) == nan,
         * should be y0 =(y-dt*v_y)
         */
        else if (vel.x == 0)
            {
            x0 = pos.x;
            z0 = (pos.z - dt * vel.z);
            y0 = sign * (A * fast::cos(x0 * m_wavenumber) + A + m_H_narrow);
            }
        else // not horizontal or vertical collision - do Newthon's method
            {
            delta = fabs(0
                         - (sign * (A * fast::cos(x0 * m_wavenumber) + A + m_H_narrow)
                            - vel.y / vel.x * (x0 - pos.x) - pos.y));

            unsigned int counter = 0;
            while (delta > target_precision && counter < max_iteration)
                {
                fast::sincos(x0 * m_wavenumber, s, c);
                n = sign * (A * c + A + m_H_narrow) - vel.y / vel.x * (x0 - pos.x) - pos.y; // f
                n2 = -sign * m_wavenumber * A * s - vel.y / vel.x;                          // df
                x0 = x0 - n / n2; // x = x - f/df
                delta = fabs(0
                             - (sign * (A * fast::cos(x0 * m_wavenumber) + A + m_H_narrow)
                                - vel.y / vel.x * (x0 - pos.x) - pos.y));
                ++counter;
                }
            /* The new z position is calculated from the wall equation to guarantee that the new
             * particle positon is exactly at the wall and not accidentally slightly inside of the
             * wall because of nummerical presicion.
             */
            y0 = sign * (A * fast::cos(x0 * m_wavenumber) + A + m_H_narrow);

            /* The new y position can be calculated from the fact that the last position outside of
             * the wall, the current position inside of the  wall, and the new position exactly at
             * the wall are on a straight line.
             */
            z0 = -(pos.x - dt * vel.x - x0) * vel.z / vel.x + (pos.z - dt * vel.z);

            // Newton's method sometimes failes to converge (close to saddle points, df'==0,
            // overshoot, bad initial guess,..) catch all of them here and do bisection if Newthon's
            // method didn't work
            Scalar lower_x = fmin(pos.x - dt * vel.x, pos.x);
            Scalar upper_x = fmax(pos.x - dt * vel.x, pos.x);

            // found intersection is NOT in between old and new point, ie intersection is
            // wrong/inaccurate. do bisection to find intersection - slower but more robust than
            // Newton's method
            if (x0 < lower_x || x0 > upper_x)
                {
                counter = 0;
                Scalar3 point1 = pos;                     // final position at t+dt
                Scalar3 point2 = pos - dt * vel;          // initial position
                Scalar3 point3 = 0.5 * (point1 + point2); // halfway point
                Scalar fpoint3 = (sign * (A * fast::cos(point3.x * m_wavenumber) + A + m_H_narrow)
                                  - point3.y); // value at halfway point, f(x)
                // Note: technically, the presicion of Newton's method and bisection is slightly
                // different, with bisection being less precise and slower convergence.
                while (fabs(fpoint3) > target_precision && counter < max_iteration)
                    {
                    fpoint3 = (sign * (A * fast::cos(point3.x * m_wavenumber) + A + m_H_narrow)
                               - point3.y);
                    // because we know that point1 outside of the channel and point2 is inside of
                    // the channel, we only need to check the halfway point3 - if it is inside,
                    // replace point2, if it is outside, replace point1
                    if (isOutside(point3) == false)
                        {
                        point2 = point3;
                        }
                    else
                        {
                        point1 = point3;
                        }
                    point3 = 0.5 * (point1 + point2);
                    ++counter;
                    }
                // final point3 == intersection
                x0 = point3.x;
                y0 = sign * (A * fast::cos(x0 * m_wavenumber) + A + m_H_narrow);
                z0 = -(pos.x - dt * vel.x - x0) * vel.z / vel.x + (pos.z - dt * vel.z);
                }
            }

        // Remaining integration time dt is amount of time spent traveling distance out of bounds.
        Scalar3 pos_new = make_scalar3(x0, y0, z0);
        dt = fast::sqrt(dot((pos - pos_new), (pos - pos_new)) / dot(vel, vel));
        pos = pos_new;

        /* update velocity according to boundary conditions.
         *
         * A upwards normal of the surface is given by (-df/dx,-df/dy,1) with f =
         * sign*(A*cos(x*2*pi*p/L)+A+h), so normal  = (sign*A*2*pi*p/L*sin(x*2*pi*p/L),0,1)/|length|
         * We define B = sign*A*2*pi*p/L*sin(x*2*pi*p/L), so then the normal is given by
         * (B,0,1)/|length| The direction of the normal is not important for the reflection.
         */
        Scalar3 vel_new;
        if (m_no_slip) // No-slip requires reflection of both tangential and normal components:
            {
            vel_new = -vel;
            }
        else // Slip conditions require only tangential components to be reflected:
            {
            Scalar B = sign * A * m_wavenumber * fast::sin(x0 * m_wavenumber);
            // The reflected vector is given by v_reflected = -2*(v_normal*v_incoming)*v_normal +
            // v_incoming Calculate components by hand to avoid sqrt in normalization of the normal
            // of the surface.
            vel_new.x = vel.x - 2 * B * (B * vel.x + vel.y) / (B * B + 1);
            vel_new.z = vel.z;
            vel_new.y = vel.y - 2 * (B * vel.x + vel.y) / (B * B + 1);
            }
        vel = vel_new;

        return true;
        }

    //! Check if a particle is out of bounds
    /*!
     * \param pos Current particle position
     * \returns True if particle is out of bounds, and false otherwise
     */
    HOSTDEVICE bool isOutside(const Scalar3& pos) const
        {
        const Scalar a = 0.5 * (m_H_wide - m_H_narrow) * fast::cos(pos.x * m_wavenumber)
                         + 0.5 * (m_H_wide - m_H_narrow) + m_H_narrow;
        return (pos.y > a || pos.y < -a);
        }

    //! Validate that the simulation box is large enough for the geometry
    /*!
     * \param box Global simulation box
     * \param cell_size Size of MPCD cell
     *
     * The box is large enough for the cosine if it is padded along the z direction so that
     * the cells just outside the highest point of the cosine would not interact with each
     * other through the boundary.
     */
    HOSTDEVICE bool validateBox(const BoxDim& box, Scalar cell_size) const
        {
        const Scalar hi = box.getHi().y;
        const Scalar lo = box.getLo().y;
        return ((hi - m_H_wide) >= cell_size && (-m_H_wide - lo) >= cell_size);
        }

    //! Get channel width at widest point
    /*!
     * \returns Channel width at widest point
     */
    HOSTDEVICE Scalar getExpansionSeparation() const
        {
        return Scalar(2.0) * m_H_wide;
        }

    //! Get channel width at narrowest point
    /*!
     * \returns Channel width at narrowest point
     */
    HOSTDEVICE Scalar getContractionSeparation() const
        {
        return Scalar(2.0) * m_H_narrow;
        }

    //! Get channel wavenumber
    HOSTDEVICE Scalar getWavenumber() const
        {
        return m_wavenumber;
        }

    //! Get the wall boundary condition
    /*!
     * \returns Boundary condition at wall
     */
    HOSTDEVICE bool getNoSlip() const
        {
        return m_no_slip;
        }

#ifndef __HIPCC__
    //! Get the unique name of this geometry
    static std::string getName()
        {
        return std::string("CosineExpansionContraction");
        }
#endif // __HIPCC__

    private:
    const Scalar m_wavenumber; //!< Wavenumber of cosine
    const Scalar m_H_wide;     //!< Half of the channel widest width
    const Scalar m_H_narrow;   //!< Half of the channel narrowest width
    const bool m_no_slip;      //!< Boundary condition
    };

    } // end namespace mpcd
    } // end namespace hoomd
#undef HOSTDEVICE

#endif // MPCD_COSINE_EXPANSION_CONTRACTION_GEOMETRY_H_
