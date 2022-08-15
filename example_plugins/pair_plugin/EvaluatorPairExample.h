// Copyright (c) 2009-2022 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_EXAMPLE_H__
#define __PAIR_EVALUATOR_EXAMPLE_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairExample.h
    \brief Defines the pair evaluator class for the example potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
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

class EvaluatorPairExample
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar a;    //!< Parameter a
        Scalar b;    //!< Parameter b

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
        param_type() : a(0), b(0) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            a = v["a"].cast<Scalar>();
            b = v["b"].cast<Scalar>();
            }

        // this constructor facilitates unit testing
        param_type(Scalar a, Scalar b, bool managed = false)
            {
            this->a = a;
            this->b = b;
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["a"] = a;
            v["b"] = b;
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
    DEVICE EvaluatorPairExample(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), a(_params.a), b(_params.b)
        {
        }

    //! Example doesn't use diameter
    DEVICE static bool needsDiameter()
        {
        return false;
        }
    //! Accept the optional diameter values
    /*! \param di Diameter of particle i
        \param dj Diameter of particle j
    */
    DEVICE void setDiameter(Scalar di, Scalar dj) { }

    //! Example doesn't use charge
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
        \note There is no need to check if rsq < rcutsq in this method.
        Cutoff tests are performed in PotentialPair.

        \return True if they are evaluated or false if they are not because
        we are beyond the cutoff
    */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
        // compute the force divided by r in force_divr
        if (rsq < rcutsq)
            {

            
            Scalar r2inv = 1 / rsq;

            Scalar r6inv = r2inv * r2inv * r2inv;
            force_divr = r2inv * r6inv * (Scalar(12.0) * a * r6inv - Scalar(6.0) * b);

            pair_eng = r6inv * (a * r6inv - b);

            if (energy_shift)
                {
                Scalar rcut2inv = Scalar(1.0) / rcutsq;
                Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;
                pair_eng -= rcut6inv * (a * rcut6inv - b);
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
        return std::string("example_pair");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;    //!< Stored rsq from the constructor
    Scalar rcutsq; //!< Stored rcutsq from the constructor
    Scalar a;    //!< a parameter extracted from the params passed to the constructor
    Scalar b;    //!< b parameter extracted from the params passed to the constructor
    // Add any additional fields
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_EXAMPLE_H__
