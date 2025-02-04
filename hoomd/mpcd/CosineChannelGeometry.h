// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CosineChannelGeometry.h
 * \brief Definition of the MPCD symmetric cosine channel geometry
 */

#ifndef MPCD_COSINE_CHANNEL_GEOMETRY_H_
#define MPCD_COSINE_CHANNEL_GEOMETRY_H_

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

//! Sinusoidal channel geometry
/*!
 * This class defines a channel with anti-symmetric cosine walls given by the
 * equations (A cos(x*2*pi*p/Lx) +/- H_narrow), creating a sinusoidal channel.
 * A is the amplitude and p is the period of the wall cosine.
 * H_narrow is the half height of the channel. This creates a wavy channel.
 * The cosine wall wavelength/frenquency needs to be consumable with the
 * periodic boundary conditions in x, therefore the period p is specified and
 * the wavelength 2*pi*p/Lx is calculated.
 *
 * Below is what the channel looks like with A=5, h=2, p=1 in a box of 10x10x18:
 *
 *    8 +------------------------------------------------+
 *      |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 *      |XXXXXXXXXXXXXXXXXXXX========XXXXXXXXXXXXXXXXXXXX|
 *    6 |XXXXXXXXXXXXXXXXXX==        ==XXXXXXXXXXXXXXXXXX|
 *      |XXXXXXXXXXXXXXXXX==          ==XXXXXXXXXXXXXXXXX|
 *      |XXXXXXXXXXXXXXX==              ==XXXXXXXXXXXXXXX|
 *    4 |XXXXXXXXXXXXXX==                ==XXXXXXXXXXXXXX|
 *      |XXXXXXXXXXXXX==                  ==XXXXXXXXXXXXX|
 *      |XXXXXXXXXXXX==      ========      ==XXXXXXXXXXXX|
 *    2 |XXXXXXXXXXX==     ===XXXXXX===     ==XXXXXXXXXXX|
 *      |XXXXXXXXXX==     ==XXXXXXXXXX==     ==XXXXXXXXXX|
 *      |XXXXXXXXX==     ==XXXXXXXXXXXX==     ==XXXXXXXXX|
 *    0 |XXXXXXXX==     =XXXXXXXXXXXXXXXX=     ==XXXXXXXX|
 * z    |XXXXXXX==     =XXXXXXXXXXXXXXXXXX=     ==XXXXXXX|
 *      |XXXXXX=      =XXXXXXXXXXXXXXXXXXXX=      =XXXXXX|
 *      |XXXX==     ==XXXXXXXXXXXXXXXXXXXXXX==     ==XXXX|
 *   -2 |XX===     ==XXXXXXXXXXXXXXXXXXXXXXXX==     ===XX|
 *      |===      ==XXXXXXXXXXXXXXXXXXXXXXXXXX==      ===|
 *      |        ==XXXXXXXXXXXXXXXXXXXXXXXXXXXX==        |
 *   -4 |       ==XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX==       |
 *      |      ==XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX==      |
 *      |     ==XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX==     |
 *   -6 |   ==XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX==   |
 *      |===XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX===|
 *      |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
 *   -8 +------------------------------------------------+
 *          -4        -2         0        2         4
 *                               x
 *
 *
 * The wall boundary conditions can optionally be changed to slip conditions.
 */
class __attribute__((visibility("default"))) CosineChannelGeometry
    {
    public:
    //! Constructor
    /*!
     * \param amplitude Amplitude of the cosine
     * \param repeat_length Repeating length of the cosine
     * \param separation Separation between channel walls
     * \param no_slip Boundary condition at the wall (slip or no-slip)
     */
    HOSTDEVICE
    CosineChannelGeometry(Scalar amplitude, Scalar repeat_length, Scalar separation, bool no_slip)
        : m_amplitude(amplitude), m_wavenumber(Scalar(2.0) * Scalar(M_PI) / repeat_length),
          m_H(Scalar(0.5) * separation), m_no_slip(no_slip)
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

        Scalar a = pos.y - m_amplitude * fast::cos(pos.x * m_wavenumber);
        const signed char sign = (char)((a > m_H) - (a < -m_H));
        // exit immediately if no collision is found
        if (sign == 0)
            {
            dt = Scalar(0);
            return false;
            }

        /* Calculate position (x0,y0,z0) of collision with wall:
         *  Because there is no analytical solution for f(x) = cos(x)-x = 0, we use Newton's method
         * to numerically estimate the x positon of the intersection first. It is convinient to use
         * the halfway point between the last particle position outside the wall (at time t-dt) and
         * the current position inside the wall (at time t) as initial guess for the intersection.
         *
         *  We limit the number of iterations (max_iteration) and the desired presicion
         * (target_precision) for performance reasons.
         */
        const unsigned int max_iteration = 6;
        const Scalar target_precision = 1e-5;

        Scalar x0 = pos.x - 0.5 * dt * vel.x;
        Scalar y0;
        Scalar z0;

        /* catch the case where a particle collides exactly vertically (v_x=0 -> old x pos = new x
         * pos) In this case, y0 = -(0)*0/0 + (y-dt*v_y) == nan, should be y0 =(y-dt*v_y)
         */
        if (vel.x == 0) // exactly vertical x-collision
            {
            x0 = pos.x;
            y0 = (m_amplitude * fast::cos(x0 * m_wavenumber) + sign * m_H);
            z0 = (pos.z - dt * vel.z);
            }
        else if (vel.y == 0) // exactly horizontal z-collision
            {
            x0 = fast::acos((pos.y - sign * m_H) / m_amplitude) / m_wavenumber;
            y0 = pos.y;
            z0 = -(pos.x - dt * vel.x - x0) * vel.z / vel.x + (pos.z - dt * vel.z);
            }
        else
            {
            Scalar delta = fabs(-((m_amplitude * fast::cos(x0 * m_wavenumber) + sign * m_H)
                                  - vel.y / vel.x * (x0 - pos.x) - pos.y));

            Scalar n, n2;
            Scalar s, c;
            unsigned int counter = 0;
            while (delta > target_precision && counter < max_iteration)
                {
                fast::sincos(x0 * m_wavenumber, s, c);
                n = (m_amplitude * c + sign * m_H) - vel.y / vel.x * (x0 - pos.x) - pos.y; // f
                n2 = -m_wavenumber * m_amplitude * s - vel.y / vel.x;                      // df
                x0 = x0 - n / n2; // x = x - f/df
                delta = fabs(-((m_amplitude * fast::cos(x0 * m_wavenumber) + sign * m_H)
                               - vel.y / vel.x * (x0 - pos.x) - pos.y));
                ++counter;
                }

            /* The new y position is calculated from the wall equation to guarantee that the new
             * particle positon is exactly at the wall and not accidentally slightly inside of the
             * wall because of nummerical presicion.
             */
            y0 = (m_amplitude * fast::cos(x0 * m_wavenumber) + sign * m_H);

            /* The new z position can be calculated from the fact that the last position outside of
             * the wall, the current position inside of the  wall, and the new position exactly at
             * the wall are on a straight line.
             */
            z0 = -(pos.x - dt * vel.x - x0) * vel.z / vel.x + (pos.z - dt * vel.z);

            // Newton's method sometimes failes to converge (close to saddle points, df'==0, bad
            // initial guess,overshoot,..) catch all of them here and do bisection if Newthon's
            // method didn't work
            const Scalar x_back = pos.x - dt * vel.x;
            Scalar lower_x = fmin(x_back, pos.x);
            Scalar upper_x = fmax(x_back, pos.x);

            // found intersection is NOT in between old and new point, ie intersection is
            // wrong/inaccurate. do bisection to find intersection - slower but more robust than
            // Newton's method
            if (x0 < lower_x || x0 > upper_x)
                {
                counter = 0;
                Scalar3 point1 = pos;            // final position at t+dt, outside of channel
                Scalar3 point2 = pos - dt * vel; // initial position, inside of channel
                Scalar3 point3 = 0.5 * (point1 + point2); // halfway point
                Scalar fpoint3 = (m_amplitude * fast::cos(point3.x * m_wavenumber) + sign * m_H)
                                 - point3.y; // value at halfway point, f(x)
                // Note: technically, the presicion of Newton's method and bisection is slightly
                // different, with bisection being less precise and slower convergence.
                while (fabs(fpoint3) > target_precision && counter < max_iteration)
                    {
                    fpoint3 = (m_amplitude * fast::cos(point3.x * m_wavenumber) + sign * m_H)
                              - point3.y;
                    // because we know that point1 outside of the channel and point2 is inside of
                    // the channel, we only need to check the halfway point3 - if it is inside,
                    // replace point2, if it is outside, replace point1
                    if (!isOutside(point3))
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
                y0 = (m_amplitude * fast::cos(x0 * m_wavenumber) + sign * m_H);
                z0 = -(pos.x - dt * vel.x - x0) * vel.z / vel.x + (pos.z - dt * vel.z);
                }
            }

        // Remaining integration time dt is amount of time spent traveling distance out of bounds.
        Scalar3 pos_new = make_scalar3(x0, y0, z0);
        dt = slow::sqrt(dot((pos - pos_new), (pos - pos_new)) / dot(vel, vel));
        pos = pos_new;

        /* update velocity according to boundary conditions.
         *
         * A upwards normal of the surface is given by (-df/dx,-df/dy,1) with f = (A*cos(x*2*pi*p/L)
         * +/- sign*h), so normal  = (A*2*pi*p/L*sin(x*2*pi*p/L),0,1)/|length| We define B =
         * A*2*pi*p/L*sin(x*2*pi*p/L), so then the normal is given by (B,0,1)/sqrt(B^2+1) The
         * direction of the normal is not important for the reflection.
         */
        Scalar3 vel_new;
        if (m_no_slip) // No-slip requires reflection of both tangential and normal components:
            {
            vel_new = -vel;
            }
        else // Slip conditions require only tangential components to be reflected:
            {
            Scalar B = m_amplitude * m_wavenumber * fast::sin(x0 * m_wavenumber);
            // The reflected vector is given by v_reflected = -2*(v_normal*v_incoming)*v_normal +
            // v_incoming Calculate components by hand to avoid sqrt in normalization of the normal
            // of the surface.
            vel_new.x = vel.x - Scalar(2.0) * B * (B * vel.x + vel.y) / (B * B + Scalar(1.0));
            vel_new.z = vel.z;
            vel_new.y = vel.y - Scalar(2.0) * (B * vel.x + vel.y) / (B * B + Scalar(1.0));
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
        Scalar a = pos.y - m_amplitude * fast::cos(m_wavenumber * pos.x);
        return (a > m_H || a < -m_H);
        }

    //! Add a contribution to random virtual particle velocity.
    /*!
     * \param vel Velocity of virtual particle
     * \param pos Position of virtual particle
     *
     * No velocity contribution is needed as the wall is stationary.
     */
    HOSTDEVICE void addToVirtualParticleVelocity(Scalar3& vel, const Scalar3& pos) const { }

    //! Get channel amplitude
    HOSTDEVICE Scalar getAmplitude() const
        {
        return m_amplitude;
        }

    //! Get channel repeat length
    HOSTDEVICE Scalar getRepeatLength() const
        {
        return Scalar(2.0) * Scalar(M_PI) / m_wavenumber;
        }

    //! Get channel separation width
    HOSTDEVICE Scalar getSeparation() const
        {
        return Scalar(2.0) * m_H;
        }

    //! Get the wall boundary condition
    HOSTDEVICE bool getNoSlip() const
        {
        return m_no_slip;
        }

#ifndef __HIPCC__
    //! Get the unique name of this geometry
    static std::string getName()
        {
        return std::string("CosineChannel");
        }
#endif // __HIPCC__

    private:
    const Scalar m_amplitude;  //!< Amplitude of the channel
    const Scalar m_wavenumber; //!< Wavenumber of cosine
    const Scalar m_H;          //!< Half of the channel separation
    const bool m_no_slip;      //!< Boundary condition
    };

    } // end namespace mpcd
    } // end namespace hoomd
#undef HOSTDEVICE

#endif // MPCD_COSINE_CHANNEL_GEOMETRY_H_
