// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ManagedArray.h"
#include <memory>

// need to declare these class methods with __device__ qualifiers when building in nvcc
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

#ifndef __HIPCC__
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#endif

#ifndef __TABLEPOTENTIAL_H__
#define __TABLEPOTENTIAL_H__

namespace hoomd
    {
namespace md
    {
//! Computes the result of a tabulated pair potential
/*! The potential and force values are provided the tables V(r) and F(r) at N_table discreet \a r
    values between \a rmin and \a rcut. Evaluations are performed by simple linear interpolation.
    F(r) must be explicitly specified as -dV/dr to avoid errors resulting from the numerical
    derivative.

    V(r) and F(r) are specified for each unique particle type pair. dr is the linear bin
    spacing and equal to (rcut - rmin)/N_table. V(0) is the value of V at r=rmin. V(i) is the value
    of V at r=rmin + dr*i where i is chosen such that r >= rmin and r < rcut. V(r) and F(r) for
    r < rmin and r >= rcut is 0.

    V and F Values are interpolated linearly between two points on either side of the given r.
*/
class EvaluatorPairTable
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        Scalar rmin;                  //!< the distance of the first index of the table potential
        ManagedArray<Scalar> V_table; //!< the tabulated energy
        ManagedArray<Scalar> F_table; //!< the tabulated force specifically - (dV / dr)

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory allocation
         */
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes)
            {
            V_table.load_shared(ptr, available_bytes);
            F_table.load_shared(ptr, available_bytes);
            }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const
            {
            V_table.allocate_shared(ptr, available_bytes);
            F_table.allocate_shared(ptr, available_bytes);
            }

#ifdef ENABLE_HIP
        //! Attach managed memory to CUDA stream
        void set_memory_hint() const
            {
            V_table.set_memory_hint();
            F_table.set_memory_hint();
            }
#endif

#ifndef __HIPCC__
        param_type() : rmin(0.0) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            const auto V_py = v["U"].cast<pybind11::array_t<Scalar>>().unchecked<1>();
            const auto F_py = v["F"].cast<pybind11::array_t<Scalar>>().unchecked<1>();

            if (V_py.size() != F_py.size())
                {
                throw std::runtime_error("The length of V and F arrays must be equal.");
                }

            if (V_py.size() == 0)
                {
                throw std::runtime_error("The length of V and F must not be zero.");
                }

            size_t width = V_py.size();
            rmin = v["r_min"].cast<Scalar>();
            V_table = ManagedArray<Scalar>(static_cast<unsigned int>(width), managed);
            F_table = ManagedArray<Scalar>(static_cast<unsigned int>(width), managed);
            std::copy(V_py.data(0), V_py.data(0) + width, V_table.get());
            std::copy(F_py.data(0), F_py.data(0) + width, F_table.get());
            }

        pybind11::dict asDict() const
            {
            const auto V = pybind11::array_t<Scalar>(V_table.size(), V_table.get());
            const auto F = pybind11::array_t<Scalar>(F_table.size(), F_table.get());
            auto params = pybind11::dict();
            params["U"] = V;
            params["F"] = F;
            params["r_min"] = rmin;
            return params;
            }
#endif
        }
#if HOOMD_LONGREAL_SIZE == 32
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairTable(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), rmin(_params.rmin), V_table(_params.V_table),
          F_table(_params.F_table)
        {
        }

    //! Table doesn't use charge
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
        \param pair_eng Output parameter to write the computed pair energy.
        \param energy_shift Table potentials do not support energy shifting.

        \return True if the force and energy are evaluated or false if r is outside the valid
        range.
    */
    DEVICE bool
    evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, const bool energy_shift) const
        {
        unsigned int width = V_table.size();

        const Scalar r = fast::sqrt(rsq);
        // compute the force divided by r in force_divr
        if (rsq >= rcutsq || r < rmin)
            {
            return false;
            }
        const Scalar rcut = fast::sqrt(rcutsq);
        const Scalar delta_r = (rcut - rmin) / static_cast<Scalar>(width);
        // precomputed term
        const Scalar value_f = (r - rmin) / delta_r;

        // compute index into the table and read in values
        unsigned int value_i = static_cast<unsigned int>(slow::floor(value_f));
        // unpack the data
        const Scalar V0 = V_table[value_i];
        const Scalar F0 = F_table[value_i];
        Scalar V1 = 0;
        Scalar F1 = 0;
        if (value_i + 1 < width)
            {
            V1 = V_table[value_i + 1];
            F1 = F_table[value_i + 1];
            }

        // compute the linear interpolation coefficient
        const Scalar f = value_f - Scalar(value_i);

        // interpolate to get V and F;
        const Scalar V = V0 + f * (V1 - V0);
        const Scalar F = F0 + f * (F1 - F0);

        // return the force divided by r
        if (rsq > Scalar(0.0))
            {
            force_divr = F / r;
            }
        pair_eng = V;
        return true;
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
        return std::string("table");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    Scalar rsq;                          //!< distance squared
    Scalar rcutsq;                       //!< the potential cuttoff distance squared
    size_t width;                        //!< the distance between table indices
    Scalar rmin;                         //!< the distance of the first index of the table potential
    const ManagedArray<Scalar>& V_table; //!< the tabulated energy
    const ManagedArray<Scalar>& F_table; //!< the tabulated force specifically - (dV / dr)
    };

    } // end namespace md
    } // end namespace hoomd

#endif
