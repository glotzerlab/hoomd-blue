// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jproc

#ifndef __ALCHEMICALPOTENTIALPAIR_H__
#define __ALCHEMICALPOTENTIALPAIR_H__

#include "hoomd/Index1D.h"
#include "hoomd/md/PotentialPair.h"
#include <bitset>

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

/*! \file AlchemicalPotentialPair.h
    \brief Defines the template class for alchemical pair potentials
    \details The heart of the code that computes pair potentials is in this file.
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

//! Template class for computing alchemical pair potentials
/*! <b>Overview:</b>

    <b>Implementation details</b>



    \sa export_PotentialPair()
*/
template<class evaluator> class AlchemicalPotentialPair : public PotentialPair<evaluator>
    {
    public:
    //! Construct the pair potential
    AlchemicalPotentialPair(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<NeighborList> nlist,
                            const std::string& log_suffix = "");
    //! Destructor
    virtual ~AlchemicalPotentialPair();

    void setNextAlchemStep(uint64_t next)
        {
        m_nextAlchemTimeStep = next;
        }
    //! Returns a list of log quantities this compute calculates
    virtual std::vector<std::string> getProvidedLogQuantities();
    //! Calculates the requested log value and returns it
    virtual Scalar getLogValue(const std::string& quantity, uint64_t timestep);

    private:
    typedef std::bitset<evaluator::num_alchemical_parameters> mask_type;

    protected:
    GlobalArray<mask_type> m_alchemy_mask; //!< Type pair mask for if alchemical forces are used
    GlobalArray<GlobalArray<Scalar>>
        m_alchemical_forces; //!< Per type pair, per particle IF used ELSE 0 sized
    Index3D m_alchemy_index; //!< type i,type j, alchemical parameter
    bool pre_alch_step;      //!< Flag for if alchemical forces need computing on this timestep
    uint64_t m_nextAlchemTimeStep; //!< Next alchemical time step

    //! Method to be called when number of types changes
    virtual void slotNumTypesChange()
        {
        Index2D new_type_pair_idx = Index2D(this->m_pdata->getNTypes());
        GlobalArray<mask_type> new_mask(new_type_pair_idx.getNumElements(), this->m_exec_conf);
        GlobalArray<GlobalArray<Scalar>> new_alchemical_forces(new_type_pair_idx.getW(),
                                                               new_type_pair_idx.getH(),
                                                               evaluator::num_alchemical_parameters,
                                                               this->m_exec_conf);

        ArrayHandle<mask_type> h_new_mask(new_mask, access_location::host, access_mode::overwrite);
        for (unsigned int i = 0; i < new_type_pair_idx.getNumElements(); i++)
            {
            h_new_mask.data[i].reset(); // set bitsets to all false by default, enable manually
            }

        ArrayHandle<mask_type> h_mask(m_alchemy_mask, access_location::host, access_mode::read);
        // copy over entries that are valid in both the new and old matrices
        unsigned int copy_w = std::min(new_type_pair_idx.getW(), this->m_typpair_idx.getW());
        unsigned int copy_h = std::min(new_type_pair_idx.getH(), this->m_typpair_idx.getH());
        for (unsigned int i = 0; i < copy_w; i++)
            {
            for (unsigned int j = 0; j < copy_h; j++)
                {
                h_new_mask.data[new_type_pair_idx(i, j)] = h_mask.data[this->m_typpair_idx(i, j)];
                }
            }

        m_alchemical_forces.swap(new_alchemical_forces);
        m_alchemy_mask.swap(new_mask);
        // don't assign new_type_pair_idx

        PotentialPair<evaluator>::slotNumTypesChange();
        allocateAlchemicalForceArrays();
        }

    virtual void allocateAlchemicalForceArrays()
        {
        ArrayHandle<mask_type> h_alchemy_mask(m_alchemy_mask,
                                              access_location::host,
                                              access_mode::read);
        ArrayHandle<GlobalArray<Scalar>> h_alchemical_forces(m_alchemical_forces,
                                                             access_location::host,
                                                             access_mode::overwrite);

        for (unsigned int i = 0; i < m_alchemy_index.getW(); i++)
            {
            for (unsigned int j = 0; j < m_alchemy_index.getH(); ++j)
                {
                for (unsigned int k = 0; k < m_alchemy_index.getD(); ++k)
                    {
                    GlobalArray<Scalar> new_array;
                    if (h_alchemy_mask.data[this->m_typpair_idx(i, j)][k])
                        {
                        GlobalArray<Scalar> new_array(this->m_pdata->getN(), this->m_exec_conf);
                        }
                    else
                        {
                        GlobalArray<Scalar> new_array(0, this->m_exec_conf);
                        }
                    h_alchemical_forces.data[m_alchemy_index(i, j, k)] = new_array;
                    }
                }
            }
        }
    // Extra steps to insert
    inline void extraPreparation(uint64_t timestep);
    inline void extraPerNeighbor(evaluator eval);
    };

template<class evaluator>
AlchemicalPotentialPair<evaluator>::AlchemicalPotentialPair(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist,
    const std::string& log_suffix)
    : PotentialPair<evaluator>(sysdef, nlist, log_suffix)
    {
    this->m_exec_conf->msg->notice(5) << "Constructing AlchemicalPotentialPair<" << evaluator::getName()
                                << ">" << std::endl;
    }

// TODO: constructor from base class

template<class evaluator> 
inline void AlchemicalPotentialPair<evaluator>::extraPreparation(uint64_t timestep)
    {
    // zero force
    if (timestep == m_nextAlchemTimeStep)
        {
        this->m_exec_conf->msg->notice(10)
            << "AlchemPotentialPair: Calculating alchemical forces" << std::endl;
        }
        // TODO: actually zero forces using the memset syntax
    }

template<class evaluator> inline void AlchemicalPotentialPair<evaluator>::extraPerNeighbor(evaluator eval)
    {
    Scalar alphas[evaluator::num_alchemical_parameters] = {1.0};
    Scalar d_alchemical[evaluator::num_alchemical_parameters] = {0.0};
    eval.AlchemParams(alphas);
    eval.evalDAlphaEnergy(d_alchemical, alphas);
    }
// per particle neighbor
// // set up alchemical values for the pair (no longer needed)
// for (uint64_t m = 0; m < m_alchem_m; m++)
//     {
//     alpha_ij[m] = Scalar(0.5) * (h_alpha.data[m*m_alchem_pitch+i].x +
//     h_alpha.data[m*m_alchem_pitch+j].x);
//     }
// }

//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialPair class template.
*/
template<class T> void export_AlchemicalPotentialPair(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<T, ForceCompute, std::shared_ptr<T>> aLchemicalpotentialpair(m, name.c_str());
    aLchemicalpotentialpair
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<NeighborList>,
                            const std::string&>())
        .def("setParams", &T::setParamsPython)
        .def("getParams", &T::getParams)
        .def("setRCut", &T::setRCutPython)
        .def("getRCut", &T::getRCut)
        .def("setROn", &T::setROnPython)
        .def("getROn", &T::getROn)
        .def_property("mode", &T::getShiftMode, &T::setShiftModePython)
        .def("computeEnergyBetweenSets", &T::computeEnergyBetweenSetsPythonList)
        .def("slotWriteGSDShapeSpec", &T::slotWriteGSDShapeSpec)
        .def("connectGSDShapeSpec", &T::connectGSDShapeSpec);
    }

#endif // __ALCHEMICALPOTENTIALPAIR_H__
