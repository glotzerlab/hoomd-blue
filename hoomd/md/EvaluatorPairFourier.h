// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_FOURIER_H__
#define __PAIR_EVALUATOR_FOURIER_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairFourier.h
    \brief Defines the pair evaluator class for potential in form of Fourier series

    \details .....
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
//! Class for evaluating the Fourier pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ.

    <b>Fourier specifics</b>

    EvaluatorPairFourier evaluates the function:
    \f[ V_{\mathrm{Fourier}}(r) = \frac{1}{r^{12}}
    + \frac{1}{r^2}\sum_{n=1}^4 [a_n cos(\frac{n \pi r}{r_{cut}})
    + b_n sin(\frac{n \pi r}{r_{cut}})] \f]

    where:
    \f[ a_1 = \sum_{n=2}^4 (-1)^n a_n cos(\frac{n \pi r}{r_{cut}}) \f]

    \f[ b_1 = \sum_{n=2}^4 n (-1)^n b_n cos(\frac{n \pi r}{r_{cut}}) \f]

    is calculated to enforce close to zero value at r_cut
*/
class EvaluatorPairFourier
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar a[3]; //!< Fourier component coefficents
        Scalar b[3]; //!< Fourier component coefficents

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        //! set CUDA memory hint
        void set_memory_hint() const { }
#endif

#ifndef __HIPCC__
        param_type()
            {
            for (int i = 0; i < 3; i++)
                {
                a[i] = 0.0;
                b[i] = 0.0;
                }
            }

        param_type(pybind11::dict v, bool managed = false)
            {
            pybind11::list py_a(v["a"]);
            pybind11::list py_b(v["b"]);

            for (int i = 0; i < 3; i++)
                {
                a[i] = pybind11::cast<Scalar>(py_a[i]);
                b[i] = pybind11::cast<Scalar>(py_b[i]);
                }
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["a"] = pybind11::make_tuple(a[0], a[1], a[2]);
            v["b"] = pybind11::make_tuple(b[0], b[1], b[2]);
            return v;
            }
#endif
        } __attribute__((aligned(16)));

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance beteen the particles
        \param _rcutsq Sqauared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */

    DEVICE EvaluatorPairFourier(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), params(_params)
        {
        }

    //! Fourier doesn't use charge
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

        \return True if they are evaluated or false if they are not because we are beyond the
       cuttoff
    */
    DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
        // compute the force divided by r in force_divr
        if (rsq < rcutsq)
            {
            Scalar half_period = fast::sqrt(rcutsq);
            Scalar period_scale = M_PI / half_period;
            Scalar r = fast::sqrt(rsq);
            Scalar x = r * period_scale;
            Scalar r1inv = Scalar(1) / r;
            Scalar r2inv = Scalar(1) / rsq;
            Scalar r3inv = r1inv * r2inv;
            Scalar r12inv = r3inv * r3inv * r3inv * r3inv;
            Scalar a1 = 0;
            Scalar b1 = 0;
            for (int i = 2; i < 5; i++)
                {
                Scalar pow_neg1_i = (i & 1) ? -1.0 : 1.0;
                a1 = a1 + pow_neg1_i * params.a[i - 2];
                b1 = b1 + i * pow_neg1_i * params.b[i - 2];
                }
            Scalar theta = x;
            Scalar s;
            Scalar c;
            fast::sincos(theta, s, c);
            Scalar fourier_part = a1 * c + b1 * s;
            force_divr = a1 * s - b1 * c;

            for (int i = 2; i < 5; i++)
                {
                theta = Scalar(i) * x;
                fast::sincos(theta, s, c);
                fourier_part += params.a[i - 2] * c + params.b[i - 2] * s;
                force_divr += params.a[i - 2] * Scalar(i) * s - params.b[i - 2] * Scalar(i) * c;
                }

            force_divr = r1inv
                         * (r1inv * r12inv * Scalar(12) + r2inv * period_scale * force_divr
                            + Scalar(2) * r3inv * fourier_part);
            pair_eng = r12inv + r2inv * fourier_part;

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
        return std::string("fourier");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;               //!< Stored rsq from the constructor
    Scalar rcutsq;            //!< Stored rcutsq from the constructor
    const param_type& params; //!< Fourier component coefficents
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_FOURIER_H__
