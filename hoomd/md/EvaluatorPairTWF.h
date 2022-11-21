// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_TWF_H__
#define __PAIR_EVALUATOR_TWF_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"
#include <cmath>

/*! \file EvaluatorPairTWF.h
    \brief Defines the pair potential evaluator for the TWF potential
    \details The potential was designed for simulating globular proteins and is
    a modification of the LJ potential with harder interactions and a variable
    well width. For more information see the Python documentation.
*/

// need to declare these class methods with __device__ qualifiers when building
// in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included
// into the host compiler
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
//! Class for evaluating the TWF pair potential
/*! <b>General Overview</b>

    EvaluatorPairTWF is a low level computation class that computes the TWF pair
    potential V(r). As the standard MD potential, it also serves as a well
    documented example of how to write additional pair potentials. "Standard"
    pair potentials in hoomd are all handled via the template class
    PotentialPair. PotentialPair takes a potential evaluator as a template
    argument. In this way, all the complicated data management and other details
    of computing the pair force and potential on every single particle is only
    written once in the template class and the difference V(r) potentials that
    can be calculated are simply handled with various evaluator classes.
    Template instantiation is equivalent to inlining code, so there is no
    performance loss.

    In hoomd, a "standard" pair potential is defined as V(rsq, rcutsq, params,
    di, dj, qi, qj), where rsq is the squared distance between the two
    particles, rcutsq is the cutoff radius at which the potential goes to 0,
    params is any number of per type-pair parameters, di, dj are the diameters
    of particles i and j, and qi, qj are the charges of particles i and j
    respectively.

    Diameter and charge are not always needed by a given pair evaluator, so it
    must provide the functions needsDiameter() and needsCharge() which return
    boolean values signifying if they need those quantities or not. A false
    return value notifies PotentialPair that it need not even load those values
    from memory, boosting performance.

    If needsDiameter() returns true, a setDiameter(Scalar di, Scalar dj) method
    will be called to set the two diameters.  Similarly, if needsCharge()
    returns true, a setCharge(Scalar qi, Scalar qj) method will be called to set
    the two charges.

    All other arguments are common among all pair potentials and passed into the
    constructor. Coefficients are handled in a special way: the pair evaluator
    class (and PotentialPair) manage only a single parameter variable for each
    type pair. Pair potentials that need more than 1 parameter can specify that
    their param_type be a compound structure and reference that. For coalesced
    read performance on G200 GPUs, it is highly recommended that param_type
    is one of the following types: Scalar, Scalar2, Scalar4.

    The program flow will proceed like this: When a potential between a pair of
    particles is to be evaluated, a PairEvaluator is instantiated, passing the
    common parameters to the constructor and calling setDiameter() and/or
    setCharge() if need be. Then, the evalForceAndEnergy() method is called to
    evaluate the force and energy (more on that later). Thus, the evaluator must
    save all of the values it needs to compute the force and energy in member
    variables.

    evalForceAndEnergy() makes the necessary computations and sets the out
    parameters with the computed values.  Specifically after the method
    completes, \a force_divr must be set to the value \f$
    -\frac{1}{r}\frac{\partial V}{\partial r}\f$ and \a pair_eng must be set to
    the value \f$ V(r) \f$ if \a energy_shift is false or \f$ V(r) -
    V(r_{\mathrm{cut}}) \f$ if \a energy_shift is true.

    A pair potential evaluator class is also used on the GPU. So all of its
    members must be declared with the DEVICE keyword before them to mark them
    __device__ when compiling in nvcc and blank otherwise. If any other code
    needs to diverge between the host and device (i.e., to use a special math
    function like __powf on the device), it can similarly be put inside an ifdef
    __HIPCC__ block.

*/
class EvaluatorPairTWF
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar sigma;
        Scalar alpha;
        Scalar prefactor;

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
        param_type() : sigma(1), alpha(1), prefactor(1) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            sigma = v["sigma"].cast<Scalar>();
            alpha = v["alpha"].cast<Scalar>();
            prefactor = 4.0 * v["epsilon"].cast<Scalar>() / (alpha * alpha);
            }

        param_type(Scalar sigma, Scalar epsilon, Scalar alpha, bool managed = false)
            : sigma(sigma), alpha(alpha), prefactor(4 * epsilon / (alpha * alpha))
            {
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["sigma"] = sigma;
            v["alpha"] = alpha;
            v["epsilon"] = alpha * alpha / 4 * prefactor;
            return v;
            }
#endif
        } __attribute__((aligned(16)));

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairTWF(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), params(_params)
        {
        }

    //! TWF doesn't use diameter
    DEVICE static bool needsDiameter()
        {
        return false;
        }

    //! Accept the optional diameter values
    /*! \param di Diameter of particle i
        \param dj Diameter of particle j
    */
    DEVICE void setDiameter(Scalar di, Scalar dj) { }

    //! TWF doesn't use charge
    DEVICE static bool needsCharge()
        {
        return false;
        }

    //! Accept the optional diameter values
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate the force and energy
    /*! \param force_divr Output parameter to write the computed force divided by r.
        \param pair_eng Output parameter to write the computed pair energy
        \param energy_shift If true, the potential must be shifted so that
        V(r) is continuous at the cutoff

        \return True if they are evaluated or false if they are not because
        we are beyond the cutoff
    */
    DEVICE bool evalForceAndEnergy(ShortReal& force_divr, ShortReal& pair_eng, bool energy_shift)
        {
        // compute the force divided by r in force_divr
        if (rsq < rcutsq)
            {
            // Compute common terms to equations
            Scalar sigma2 = params.sigma * params.sigma;
            // If particles are overlapping return maximum possible energy
            // and force since this is an invalid state and should be
            // infinite energy and force.
            if (rsq <= sigma2)
                {
                // Since std::numeric_limit<>::max cannot be used for GPU
                // code, we use the INFINITY macros instead.
                pair_eng = INFINITY;
                force_divr = INFINITY;
                return true;
                }

            Scalar common_term = 1.0 / (rsq / sigma2 - 1.0);
            Scalar common_term3 = common_term * common_term * common_term;
            Scalar common_term6 = common_term3 * common_term3;
            // Compute force and energy
            pair_eng = params.prefactor * (common_term6 - params.alpha * common_term3);
            // The force term is -(dE / dr) * (1 / r).
            Scalar force_term = 6 * common_term / sigma2;
            force_divr
                = params.prefactor * force_term * (2 * common_term6 - params.alpha * common_term3);

            if (energy_shift)
                {
                Scalar common_term_shift = 1.0 / (sigma2 / rcutsq - 1.0);
                Scalar common_term3_shift
                    = common_term_shift * common_term_shift * common_term_shift;
                Scalar common_term6_shift = common_term3_shift * common_term3_shift;
                pair_eng
                    -= params.prefactor * (common_term6_shift - params.alpha * common_term3_shift);
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
        return std::string("twf");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;        //!< Stored rsq from the constructor
    Scalar rcutsq;     //!< Stored rcutsq from the constructor
    param_type params; //!< parameters passed to the constructor
    };

    }  // end namespace md
    }  // end namespace hoomd
#endif // __PAIR_EVALUATOR_TWF_H__
