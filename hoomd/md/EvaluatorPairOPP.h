#ifndef __PAIR_EVALUATOR_OPP_H__
#define __PAIR_EVALUATOR_OPP_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairOPP.h
    \brief Defines the pair evaluator class for OPP potential
*/

// need to declare these class methods with __device__ qualifiers when building
// in nvcc DEVICE is __host__ __device__ when included in nvcc and blank when
// included into the host compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the oscillating pair potential
/*! <b>General Overview</b>

    <b>OPP specifics</b>

    EvaluatorPairOPP evaluates the function:
    \f{equation*}
    V_{\mathrm{OPP}}(r)  = - \frac{1}{r^{15}} + \frac{1}{r^{3}}
                             \cos(k ( r - 1.25) - \phi)
    \f}

*/
class EvaluatorPairOPP
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        struct param_type
            {
            Scalar k;
            Scalar phi;

            #ifdef ENABLE_HIP
            //! Set CUDA memory hints
            void set_memory_hint() const
                {
                // default implementation does nothing
                }
            #endif

            #ifndef __HIPCC__
            param_type() : k(0), phi(0) {}

            param_type(pybind11::dict v)
                {
                k = v["k"].cast<Scalar>();
                phi = v["phi"].cast<Scalar>();
                }

            pybind11::dict asDict()
                {
                pybind11::dict v;
                v["k"] = k;
                v["phi"] = phi;
                return v;
                }
            #endif
            }
            #ifdef SINGLE_PRECISION
            __attribute__((aligned(8)));
            #else
            __attribute__((aligned(16)));
            #endif

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance between the particles
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairOPP(Scalar _rsq,
                                Scalar _rcutsq,
                                const param_type& _params)
            : rsq(_rsq),
              rcutsq(_rcutsq),
              k(_params.k),
              phi(_params.phi)
            {
            }

        //! OPP does not use diameter
        DEVICE static bool needsDiameter() { return false; }

        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) {}

        //! OPP doesn't use charge
        DEVICE static bool needsCharge() { return false; }

        //! Accept the optional diameter values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }

        //! Evaluate the force and energy
        /*! \param force_divr Output parameter to write the computed force
         * divided by r.
         *  \param pair_eng Output parameter to write the computed pair energy
         *  \param energy_shift If true, the potential must be shifted so that
         *      V(r) is continuous at the cutoff

         *  \return True if they are evaluated or false if they are not because
         *  we are beyond the cutoff
         */
        DEVICE bool evalForceAndEnergy(
            Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
            {
            if (rsq < rcutsq)
                {
                // Get quantities need for both energy and force calculation
                Scalar r(fast::sqrt(rsq));
                Scalar eval_sin, eval_cos;
                fast::sincos(k * (r - 1.25) - phi, eval_sin, eval_cos);

                // Compute energy
                Scalar inv_r_3(fast::pow(rsq, -1.5));
                Scalar inv_r_15 = inv_r_3 * inv_r_3 * inv_r_3
                                  * inv_r_3 * inv_r_3;
                pair_eng = inv_r_15 - (inv_r_3 * eval_cos);

                // Compute force
                Scalar inv_r_4 = 1 / (rsq * rsq);
                Scalar inv_r_16 = 1 / (inv_r_4 * inv_r_4 * inv_r_4 * inv_r_4);
                force_divr = (k * eval_sin * inv_r_3)
                             - (3 * eval_cos * inv_r_4) - (15 * inv_r_16);
                if (energy_shift)
                    {
                    Scalar r_cut(fast::sqrt(rcutsq));

                    // Compute energy
                    Scalar inv_r_cut_3(fast::pow(rsq, -1.5));
                    Scalar inv_r_cut_15 = inv_r_cut_3 * inv_r_cut_3
                                          * inv_r_cut_3 * inv_r_cut_3
                                          * inv_r_cut_3;
                    pair_eng -= inv_r_cut_15 - 
                        (inv_r_cut_3 * fast::cos(k * (r_cut - 1.25) - phi));
                    }

                return true;
                }
            else
                {
                return false;
                }
            }

        #ifndef __HIPCC__
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as
         * this is the name energies will be logged as via analyze.log.
        */
        static std::string getName()
            {
            return std::string("opp");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error(
                "Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar k;       //!< frequency term in potential
        Scalar phi;     //!< phase shift in potential
    };


#endif // __PAIR_EVALUATOR_OPP_H__
