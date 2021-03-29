// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/ForceCompute.h"
#include "NeighborList.h"
#include "hoomd/Index1D.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/ManagedArray.h"
#include <memory>

/*! \file TablePotential.h
    \brief Declares the TablePotential class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

// need to declare these class methods with __device__ qualifiers when building in nvcc
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifndef __TABLEPOTENTIAL_H__
#define __TABLEPOTENTIAL_H__

//! Computes the potential and force on each particle based on values given in a table
/*! \b Overview
    Pair potentials and forces are evaluated for all particle pairs in the system within the given cutoff distances.
    Both the potentials and forces** are provided the tables V(r) and F(r) at discreet \a r values between \a rmin and
    \a rmax. Evaluations are performed by simple linear interpolation, thus why F(r) must be explicitly specified to
    avoid large errors resulting from the numerical derivative. Note that F(r) should store - dV/dr.

    \b Table memory layout

    V(r) and F(r) are specified for each unique particle type pair. Three parameters need to be stored for each 
    potential: rmin, rmax, and dr, the minimum r, maximum r, and spacing between r values in the table respectively. 
    V(0) is the value of V at r=rmin. V(i) is the value of V at r=rmin + dr * i where i is chosen such that r >= rmin
    and r <= rmax. V(r) for r < rmin and > rmax is 0. The same goes for F. Thus V and F are defined between the region
    [rmin,rmax], inclusive.

    \b Interpolation
    Values are interpolated linearly between two points straddling the given r. For a given r, the first point needed, i
    can be calculated via i = floorf((r - rmin) / dr). The fraction between ri and ri+1 can be calculated via
    f = (r - rmin) / dr - Scalar(i). And the linear interpolation can then be performed via V(r) ~= Vi + f * (Vi+1 - Vi)
    \ingroup computes
*/
class EvaluatorPairTable
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        struct param_type
            {
            unsigned int width;
            Scalar rmin;
            ManagedArray<Scalar> V_table;
            ManagedArray<Scalar> F_table;

            #ifdef ENABLE_HIP
            //! Set CUDA memory hints
            void set_memory_hint() const
                {
                // default implementation does nothing
                }
            #endif

            #ifndef __HIPCC__
            param_type() : width(0), rmin(0.0), V_table({}), F_table({}) {}

            param_type(pybind11::dict v)
                {
                width = v["width"].cast<unsigned int>();
                rmin = v["r_min"].cast<Scalar>();
                unsigned int align_size = 8; //for AVX
                unsigned int N_align =((width + align_size - 1)/align_size)*align_size;
                V_table = ManagedArray<Scalar>(N_align, false, 32); // 32byte alignment for AVX
                F_table = ManagedArray<Scalar>(N_align, false, 32);
                auto V_py = v["V"].cast<pybind11::array_t<Scalar>>().unchecked<1>();
                auto F_py = v["F"].cast<pybind11::array_t<Scalar>>().unchecked<1>();
                for (unsigned int i = 0; i < width; i++)
                    {
                    V_table[i] = V_py[i];
                    F_table[i] = F_py[i];
                    }
                if (V_table.size() != F_table.size())
                    {throw std::runtime_error("The length of V and F arrays must be equal");}
                }

            pybind11::dict asDict()
                {
                Scalar *V = new Scalar[width];
                Scalar *F = new Scalar[width];
                for (unsigned int i = 0; i < width; i++)
                    {
                    V[i] = (Scalar) V_table[i];
                    F[i] = (Scalar) F_table[i];
                    }

                pybind11::capsule free_V(V, [](void *f)
                    {
                    Scalar *V = reinterpret_cast<Scalar *>(f);
                    delete[] V;
                    });
                pybind11::capsule free_F(F, [](void *f)
                    {
                    Scalar *F = reinterpret_cast<Scalar *>(f);
                    delete[] F;
                    });

                pybind11::dict v;
                v["V"] = pybind11::array_t<Scalar>(pybind11::array::ShapeContainer({width,}),
                                                   pybind11::array::StridesContainer({1*sizeof(Scalar),}),
                                                   V,
                                                   free_V);
                v["F"] = pybind11::array_t<Scalar>(pybind11::array::ShapeContainer({width,}),
                                                   pybind11::array::StridesContainer({1*sizeof(Scalar),}),
                                                   F,
                                                   free_F);
                v["width"] = width;
                v["r_min"] = rmin;
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
        DEVICE EvaluatorPairTable(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), width(_params.width), rmin(_params.rmin), V_table(_params.V_table), F_table(_params.F_table)
            {
            }

        //! Table doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! Table doesn't use charge
        DEVICE static bool needsCharge() { return false; }
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
            if (rsq < rcutsq && std::sqrt(rsq) >= rmin)
                {
                Scalar delta_r = (std::sqrt(rcutsq) - rmin) / width;
                // precomputed term
                Scalar value_f = (std::sqrt(rcutsq) - std::sqrt(rsq)) / delta_r;

                // compute index into the table and read in values
                unsigned int value_i = (unsigned int)floor(value_f);
                // unpack the data
                Scalar V0 = V_table[value_i];
                Scalar V1 = V_table[value_i + 1];
                Scalar F0 = F_table[value_i];
                Scalar F1 = F_table[value_i + 1];

                // compute the linear interpolation coefficient
                Scalar f = value_f - Scalar(value_i);

                // interpolate to get V and F;
                Scalar V = V0 + f * (V1 - V0);
                Scalar F = F0 + f * (F1 - F0);

                // convert to standard variables used by the other pair computes in HOOMD-blue
                if (rsq > Scalar(0.0))
                    force_divr = F / std::sqrt(rsq);
                pair_eng = Scalar(0.5) * V;
                return true;
                }
            else
                return false;
            }

        #ifndef __HIPCC__
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
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
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar width;   //!< extracted from the params passed to the constructor
        Scalar rmin;
        ManagedArray<Scalar> V_table; //!< extracted from the params passed to the constructor
        ManagedArray<Scalar> F_table; //!< extracted from the params passed to the constructor
    };

#endif
