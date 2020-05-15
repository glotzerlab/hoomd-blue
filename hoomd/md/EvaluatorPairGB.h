// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#ifndef __EVALUATOR_PAIR_GB_H__
#define __EVALUATOR_PAIR_GB_H__

#ifndef NVCC
#include <string>
#endif

#define HOOMD_GB_MIN(i,j) ((i > j) ? j : i)
#define HOOMD_GB_MAX(i,j) ((i > j) ? i : j)

#include "hoomd/VectorMath.h"

/*! \file EvaluatorPairGB.h
    \brief Defines a an evaluator class for the Gay-Berne potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! HOSTDEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

struct pair_gb_params
    {
    Scalar epsilon;   //! The energy scale.
    Scalar lperp;     //! The semiaxis length perpendicular to the particle orientation.
    Scalar lpar;      //! The semiaxis length parallel to the particle orientation.

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const
        {
        // No-op for this struct since it contains no arrays.
        }
    };


// Nullary structure required by AnisoPotentialPair.
struct gb_shape_params
    {
    HOSTDEVICE gb_shape_params() {}

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const {}

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const {}
    #endif
    };

/*!
 * Gay-Berne potential as formulated by Allen and Germano,
 * with shape-independent energy parameter, for identical uniaxial particles.
 */

class EvaluatorPairGB
    {
    public:
        typedef pair_gb_params param_type;
        typedef gb_shape_params shape_param_type;

        //! Constructs the pair potential evaluator
        /*! \param _dr Displacement vector between particle centers of mass
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _q_i Quaternion of i^th particle
            \param _q_j Quaternion of j^th particle
            \param _params Per type pair parameters of this potential
        */
        HOSTDEVICE EvaluatorPairGB(const Scalar3& _dr,
                               const Scalar4& _qi,
                               const Scalar4& _qj,
                               const Scalar _rcutsq,
                               const param_type& _params)
            : dr(_dr),rcutsq(_rcutsq),qi(_qi),qj(_qj),
              params(_params)
            {
            }

        //! uses diameter
        HOSTDEVICE static bool needsDiameter()
            {
            return false;
            }

        //! Whether the pair potential uses shape.
        HOSTDEVICE static bool needsShape()
            {
            return false;
            }

        //! Whether the pair potential needs particle tags.
        HOSTDEVICE static bool needsTags()
            {
            return false;
            }

        //! whether pair potential requires charges
        HOSTDEVICE static bool needsCharge( )
            {
            return false;
            }

        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        HOSTDEVICE void setDiameter(Scalar di, Scalar dj){}

        //! Accept the optional shape values
        /*! \param shape_i Shape of particle i
            \param shape_j Shape of particle j
        */
        HOSTDEVICE void setShape(const shape_param_type *shapei, const shape_param_type *shapej) {}

        //! Accept the optional tags
        /*! \param tag_i Tag of particle i
            \param tag_j Tag of particle j
        */
        HOSTDEVICE void setTags(unsigned int tagi, unsigned int tagj) {}

        //! Accept the optional charge values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        HOSTDEVICE void setCharge(Scalar qi, Scalar qj){}

        //! Evaluate the force and energy
        /*! \param force Output parameter to write the computed force.
            \param pair_eng Output parameter to write the computed pair energy.
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff.
            \param torque_i The torque exerted on the i^th particle.
            \param torque_j The torque exerted on the j^th particle.
            \return True if they are evaluated or false if they are not because we are beyond the cutoff.
        */
        HOSTDEVICE  bool
        evaluate(Scalar3& force, Scalar& pair_eng, bool energy_shift, Scalar3& torque_i, Scalar3& torque_j)
            {
            Scalar rsq = dot(dr,dr);
            Scalar r = fast::sqrt(rsq);
            vec3<Scalar> unitr = fast::rsqrt(dot(dr,dr))*dr;

            // obtain rotation matrices (space->body)
            rotmat3<Scalar> rotA(conj(qi));
            rotmat3<Scalar> rotB(conj(qj));

            // last row of rotation matrix
            vec3<Scalar> a3 = rotA.row2;
            vec3<Scalar> b3 = rotB.row2;

            Scalar ca = dot(a3,unitr);
            Scalar cb = dot(b3,unitr);
            Scalar cab = dot(a3,b3);
            Scalar lperpsq = params.lperp*params.lperp;
            Scalar lparsq = params.lpar*params.lpar;
            Scalar chi=(lparsq - lperpsq)/(lparsq+lperpsq);
            Scalar chic = chi*cab;

            Scalar chi_fact = chi/(Scalar(1.0)-chic*chic);
            vec3<Scalar> kappa = Scalar(1.0/2.0)*r/lperpsq
                *(unitr - chi_fact*((ca-chic*cb)*a3+(cb-chic*ca)*b3));

            Scalar phi = Scalar(1.0/2.0)*dot(dr, kappa)/rsq;
            Scalar sigma = fast::rsqrt(phi);

            Scalar sigma_min = Scalar(2.0)*HOOMD_GB_MIN(params.lperp,params.lpar);

            Scalar zeta = (r-sigma+sigma_min)/sigma_min;
            Scalar zetasq = zeta*zeta;

            Scalar rcut = fast::sqrt(rcutsq);
            Scalar dUdphi,dUdr;

            // define r_cut to be along the long axis
            Scalar sigma_max = Scalar(2.0)*HOOMD_GB_MAX(params.lperp,params.lpar);
            Scalar zetacut = rcut/sigma_max;
            Scalar zetacutsq = zetacut*zetacut;

            // compute the force divided by r in force_divr
            if (zetasq < zetacutsq && params.epsilon != Scalar(0.0))
                {
                Scalar zeta2inv = Scalar(1.0)/zetasq;
                Scalar zeta6inv = zeta2inv * zeta2inv *zeta2inv;

                dUdr  = -Scalar(24.0)*params.epsilon*(zeta6inv/zeta*(Scalar(2.0)*zeta6inv-Scalar(1.0)))/sigma_min;
                dUdphi = dUdr*Scalar(1.0/2.0)*sigma*sigma*sigma;

                pair_eng = Scalar(4.0)*params.epsilon*zeta6inv * (zeta6inv - Scalar(1.0));

                if (energy_shift)
                    {
                    Scalar zetacut2inv = Scalar(1.0)/zetacutsq;
                    Scalar zetacut6inv = zetacut2inv * zetacut2inv * zetacut2inv;
                    pair_eng -= Scalar(4.0)*params.epsilon*zetacut6inv * (zetacut6inv - Scalar(1.0));
                    }
                }
            else
                return false;

            // compute vector force and torque
            Scalar r2inv = Scalar(1.0)/rsq;
            vec3<Scalar> fK = -r2inv*dUdphi*kappa;
            vec3<Scalar> f = -dUdr*unitr + fK + r2inv*dUdphi*unitr*dot(kappa,unitr);
            force = vec_to_scalar3(f);

            vec3<Scalar> rca = Scalar(1.0/2.0)* (-dr - r*chi_fact*((ca-chic*cb)*a3-(cb-chic*ca)*b3));
            vec3<Scalar> rcb = rca + dr;
            torque_i = vec_to_scalar3(cross(rca, fK));
            torque_j = -vec_to_scalar3(cross(rcb, fK));

            return true;
            }

        #ifndef NVCC
        //! Get the name of the potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return "gb";
            }

        std::string getShapeSpec() const
            {
            std::ostringstream shapedef;
            shapedef << "{\"type\": \"Ellipsoid\", \"a\": " << params.lperp <<
                        ", \"b\": " << params.lperp <<
                        ", \"c\": " << params.lpar <<
                        "}";
            return shapedef.str();
            }
        #endif

    protected:
        vec3<Scalar> dr;   //!< Stored dr from the constructor
        Scalar rcutsq;     //!< Stored rcutsq from the constructor
        quat<Scalar> qi;   //!< Orientation quaternion for particle i
        quat<Scalar> qj;   //!< Orientation quaternion for particle j
        const param_type &params;  //!< The pair potential parameters
    };


#undef HOOMD_GB_MIN
#undef HOOMD_GB_MAX
#endif // __EVALUATOR_PAIR_GB_H__
