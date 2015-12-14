/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
the University of Michigan All rights reserved.

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

#ifndef __EVALUATOR_CONSTRAINT_Ellipsoid_H__
#define __EVALUATOR_CONSTRAINT_Ellipsoid_H__

#include "HOOMDMath.h"
using namespace std;

/*! \file EvaluatorConstraintEllipsoid.h
    \brief Defines the constraint evaluator class for ellipsoids
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating ellipsoid constraints
/*! <b>General Overview</b>
    EvaluatorConstraintEllipsoid is a low level computation helper class to aid in evaluating particle constraints on a
    ellipsoid. Given a ellipsoid at a given position and radii, it will find the nearest point on the Ellipsoid to a given
    position.
*/
class EvaluatorConstraintEllipsoid
    {
    public:
        //! Constructs the constraint evaluator
        /*! \param _P Position of the ellipsoid
            \param _rx   Radius of the ellipsoid in the X direction
            \param _ry   Radius of the ellipsoid in the Y direction
            \param _rz   Radius of the ellipsoid in the Z direction

            NOTE: For the algorithm to work, we must have _rx >= _rz, ry >= _rz, and _rz > 0.
        */
        DEVICE EvaluatorConstraintEllipsoid(Scalar3 _P, Scalar _rx, Scalar _ry, Scalar _rz)
            : P(_P), rx(_rx), ry(_ry), rz(_rz)
            {
            }

        //! Evaluate the closest point on the ellipsoid. Method from: http://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf
        /*! \param U unconstrained point

            \return Nearest point on the ellipsoid
        */
        DEVICE Scalar3 evalClosest(const Scalar3& U)
        {
            if (rx==ry && ry==rz) // if ellipsoid is actually a sphere, use easier method
            {
                // compute the vector pointing from P to V
                Scalar3 V;
                V.x = U.x - P.x;
                V.y = U.y - P.y;
                V.z = U.z - P.z;

                // compute 1/magnitude of V
                Scalar magVinv = fast::rsqrt(V.x*V.x + V.y*V.y + V.z*V.z);

                // compute Vhat, the unit vector pointing in the direction of V
                Scalar3 Vhat;
                Vhat.x = magVinv * V.x;
                Vhat.y = magVinv * V.y;
                Vhat.z = magVinv * V.z;

                // compute resulting constrained point
                Scalar3 C;
                C.x = P.x + Vhat.x * rx;
                C.y = P.y + Vhat.y * rx;
                C.z = P.z + Vhat.z * rx;

                return C;
            }

            else // else use iterative method
            {
                Scalar xsign, ysign, zsign; // sign of point's position
                Scalar y0, y1, y2;
                if (U.x < 0) { xsign = -1; } else { xsign = 1; }
                if (U.y < 0) { ysign = -1; } else { ysign = 1; }
                if (U.z < 0) { zsign = -1; } else { zsign = 1; }
                y0 = U.x * xsign;
                y1 = U.y * ysign;
                y2 = U.z * zsign;

                Scalar z0 = y0 / rx;
                Scalar z1 = y1 / ry;
                Scalar z2 = y2 / rz;
                Scalar g = z0*z0 + z1*z1 + z2*z2 - 1;
                if (g != 0) // point does not lay on ellipsoid
                {
                    Scalar r0 = (rx/rz) * (rx/rz);
                    Scalar r1 = (ry/rz) * (ry/rz);

                    // iterative method of calculus to find closest point
                    Scalar n0 = r0 * z0;
                    Scalar n1 = r1 * z1;
                    Scalar s0 = z1 - 1;
                    Scalar s1 = 0;
                    if (g > 0) { s1 = sqrt(n0*n0 + n1*n1 + z2*z2) - 1; }
                    Scalar sbar;
                    Scalar ratio0, ratio1, ratio2;
                    int i, imax = 10000; // When tested, on average takes ~30 steps to complete, and never more than 150.
                    for (i = 0; i < imax; i++)
                    {
                        sbar = (s0 + s1) / 2.0;
                        if (sbar == s0 || sbar == s1) { break; }
                        ratio0 = n0 / (sbar + r0);
                        ratio1 = n1 / (sbar + r1);
                        ratio2 = z2 / (sbar + 1);
                        g = ratio0*ratio0 + ratio1*ratio1 + ratio2*ratio2 - 1;
                        if (g > 0) { s0 = sbar; } else if (g < 0) { s1 = sbar; } else { break; }
                    }
                    if (i == imax)
                    {
                        throw runtime_error("constrain.ellipsoid: Not enough iteration steps to find closest point on ellipsoid.\n");
                    }

                    // compute resulting constrained point
                    Scalar3 C;
                    C.x = xsign * r0 * y0 / (sbar + r0);
                    C.y = ysign * r1 * y1 / (sbar + r1);
                    C.z = zsign * y2 / (sbar + 1);

                    return C;
                }
                else // trivial case of point laying on ellipsoid
                {
                    return U; 
                }
            }
        }

        //! Evaluate the normal unit vector for point on the ellipsoid.
        /*! \param U point on ellipsoid
            \return normal unit vector for  point on the ellipsoid
        */
        DEVICE Scalar3 evalNormal(const Scalar3& U)
        {
            Scalar3 N;
            N.x = U.x / (rx*rx);
            N.y = U.y / (ry*ry);
            N.z = U.z / (rz*rz);

            Scalar nNorm;
            nNorm = sqrt(N.x*N.x + N.y*N.y + N.z*N.z);
            N.x /= nNorm;
            N.y /= nNorm;
            N.z /= nNorm;

            return N;
        }

    protected:
        Scalar3 P;      //!< Position of the ellipsoid
        Scalar rx;       //!< radius of the ellipsoid in the X direction
        Scalar ry;       //!< radius of the ellipsoid in the Y direction
        Scalar rz;       //!< radius of the ellipsoid in the Z direction
    };


#endif // __PAIR_EVALUATOR_LJ_H__
