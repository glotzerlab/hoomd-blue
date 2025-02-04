// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_DLVO_H__
#define __PAIR_EVALUATOR_DLVO_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairDLVO.h
    \brief Defines the pair evaluator class for DLVO potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
//! compiler
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd
    {
namespace md
    {
//! Class for evaluating the DLVO pair potential
/** See Israelachvili 2011, pp. 317.
 */
class EvaluatorPairDLVO
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar kappa;
        Scalar Z;
        Scalar A;
        Scalar a1;
        Scalar a2;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        //! Set CUDA memory hints
        void set_memory_hint() const
            {
            // default implementation does nothing
            }
#endif

#ifndef __HIPCC__
        param_type() : kappa(0), Z(0), A(0) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            kappa = v["kappa"].cast<Scalar>();
            Z = v["Z"].cast<Scalar>();
            A = v["A"].cast<Scalar>();
            a1 = v["a1"].cast<Scalar>();
            a2 = v["a2"].cast<Scalar>();
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["kappa"] = kappa;
            v["Z"] = Z;
            v["A"] = A;
            v["a1"] = a1;
            v["a2"] = a2;
            return v;
            }
#endif
        } __attribute__((aligned(16)));

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairDLVO(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), kappa(_params.kappa), Z(_params.Z), A(_params.A)
        {
        radsum = _params.a1 + _params.a2;
        radsub = _params.a1 - _params.a2;
        radprod = _params.a1 * _params.a2;
        radsumsq = _params.a1 * _params.a1 + _params.a2 * _params.a2;
        radsubsq = _params.a1 * _params.a1 - _params.a2 * _params.a2;
        delta = radsum - Scalar(1.0);
        }

    //! DLVO doesn't use charge
    DEVICE static bool needsCharge()
        {
        return false;
        }
    //! Accept the optional charge values.
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate the force and energy
    /*! \param force_divr Output parameter to write the computed force divided by r.
        \param pair_eng Output parameter to write the computed pair energy
        \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the
       cutoff \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are
       performed in PotentialPair.

        \return True if they are evaluated or false if they are not because we are beyond the cutoff
    */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
        // precompute some quantities
        Scalar rinv = fast::rsqrt(rsq);
        Scalar r = Scalar(1.0) / rinv;
        Scalar rcutinv = fast::rsqrt(rcutsq);
        Scalar rcut = Scalar(1.0) / rcutinv;

        // compute the force divided by r in force_divr
        if (r < rcut && kappa != 0)
            {
            Scalar rmds = r - radsum;
            Scalar rmdsqs = r * r - radsum * radsum;
            Scalar rmdsqm = r * r - radsub * radsub;
            Scalar radsuminv = Scalar(1.0) / radsum;
            Scalar rmdsqsinv = Scalar(1.0) / rmdsqs;
            Scalar rmdsqminv = Scalar(1.0) / rmdsqm;
            Scalar exp_val = fast::exp(-kappa * rmds);
            Scalar forcerep_divr = kappa * radprod * radsuminv * Z * exp_val / r;
            Scalar fatrterm1 = r * r * r * r + radsubsq * radsubsq - Scalar(2.0) * r * r * radsumsq;
            Scalar fatrterm1inv = Scalar(1.0) / fatrterm1 * Scalar(1.0) / fatrterm1;
            Scalar forceatr_divr
                = -Scalar(32.0) * A / Scalar(3.0) * radprod * radprod * radprod * fatrterm1inv;
            force_divr = forcerep_divr + forceatr_divr;

            Scalar engt1 = radprod * rmdsqsinv * A / Scalar(3.0);
            Scalar engt2 = radprod * rmdsqminv * A / Scalar(3.0);
            Scalar engt3 = slow::log(rmdsqs * rmdsqminv) * A / Scalar(6.0);
            pair_eng = r * forcerep_divr / kappa - engt1 - engt2 - engt3;
            if (energy_shift)
                {
                Scalar rcutt = rcut;
                Scalar rmdscut = rcutt - radsum;
                Scalar rmdsqscut = rcutt * rcutt - radsum * radsum;
                Scalar rmdsqmcut = rcutt * rcutt - radsub * radsub;
                Scalar rmdsqsinvcut = Scalar(1.0) / rmdsqscut;
                Scalar rmdsqminvcut = Scalar(1.0) / rmdsqmcut;

                Scalar engt1cut = radprod * rmdsqsinvcut * A / Scalar(3.0);
                Scalar engt2cut = radprod * rmdsqminvcut * A / Scalar(3.0);
                Scalar engt3cut = slow::log(rmdsqscut * rmdsqminvcut) * A / Scalar(6.0);
                Scalar exp_valcut = fast::exp(-kappa * rmdscut);
                Scalar forcerepcut_divr = kappa * radprod * radsuminv * Z * exp_valcut / rcutt;
                pair_eng -= rcutt * forcerepcut_divr / kappa - engt1cut - engt2cut - engt3cut;
                }
            return true;
            }
        else
            return false;
        }

    DEVICE Scalar evalPressureLRCIntegral()
        {
        return 0;
        }

    DEVICE Scalar evalEnergyLRCIntegral()
        {
        return 0;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("dlvo");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;      //!< Stored rsq from the constructor
    Scalar rcutsq;   //!< Stored rcutsq from the constructor
    Scalar kappa;    //!< kappa parameter extracted from the params passed to the constructor
    Scalar Z;        //!< Z parameter extracted from the params passed to the constructor
    Scalar A;        //!< A parameter extracted from the params passed to the constructor
    Scalar radsum;   //!< radsum parameter extracted from the call to setDiameter
    Scalar radsub;   //!< radsub parameter extracted from the call to setDiameter
    Scalar radprod;  //!< radprod parameter extracted from the call to setDiameter
    Scalar radsumsq; //!< radsumsq parameter extracted from the call to setDiameter
    Scalar radsubsq; //!< radsubsq parameter extracted from the call to setDiameter
    Scalar delta;    //!< Diameter sum minus one
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_DLVO_H__
