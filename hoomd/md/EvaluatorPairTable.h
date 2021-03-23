// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/ForceCompute.h"
#include "NeighborList.h"
#include "hoomd/Index1D.h"
#include "hoomd/GlobalArray.h"

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

    V(r) and F(r) are specified for each unique particle type pair. They will be indexed so that values of increasing r
    go along the rows in memory for cache efficiency when reading values. The row index to put the potential at can be
    determined using an Index2DUpperTriangular (typei, typej), as it will uniquely index each unique pair.

    To improve cache coherency even further, values for V and F will be interleaved like so: V1 F1 V2 F2 V3 F3 ... To
    accomplish this, tables are stored with a value type of Scalar2, elem.x will be V and elem.y will be F. Since Fn,
    Vn+1 and Fn+1 are read right after Vn, these are likely to be cache hits. Furthermore, on the GPU a single Scalar2
    texture read can be used to access Vn and Fn.

    Three parameters need to be stored for each potential: rmin, rmax, and dr, the minimum r, maximum r, and spacing
    between r values in the table respectively. For simple access on the GPU, these will be stored in a Scalar4 where
    x is rmin, y is rmax, and z is dr. They are indexed with the same Index2DUpperTriangular as the tables themselves.

    V(0) is the value of V at r=rmin. V(i) is the value of V at r=rmin + dr * i where i is chosen such that r >= rmin
    and r <= rmax. V(r) for r < rmin and > rmax is 0. The same goes for F. Thus V and F are defined between the region
    [rmin,rmax], inclusive.

    For ease of storing the data, all tables must be of the same number of points for all type pairs.

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
            std::vector<Scalar> V_table;
            std::vector<Scalar> F_table;

            #ifdef ENABLE_HIP
            //! Set CUDA memory hints
            void set_memory_hint() const
                {
                // default implementation does nothing
                }
            #endif

            #ifndef __HIPCC__
            param_type() : width(0), V_table({}), F_table({}) {}

            param_type(pybind11::dict v)
                {
                width = v["width"].cast<unsigned int>();
                V_table = v["V"].cast<std::vector<Scalar>>();
                F_table = v["F"].cast<std::vector<Scalar>>();
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
                v["V"] = pybind11::array_t<Scalar>(pybind11::array::ShapeContainer({width, 1}),
                                                   pybind11::array::StridesContainer({1*sizeof(Scalar), sizeof(Scalar)}),
                                                   V,
                                                   free_V);
                v["F"] = pybind11::array_t<Scalar>(pybind11::array::ShapeContainer({width, 1}),
                                                   pybind11::array::StridesContainer({1*sizeof(Scalar), sizeof(Scalar)}),
                                                   F,
                                                   free_F);
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
            :
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
    };

#endif
