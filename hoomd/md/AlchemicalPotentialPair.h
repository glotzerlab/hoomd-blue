// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jproc

#ifndef __ALCHEMICALPOTENTIALPAIR_H__
#define __ALCHEMICALPOTENTIALPAIR_H__

#include "hoomd/AlchemyData.h"
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

template<typename mask_type> struct AlchemyPackage
    {
    bool calculate_derivatives;
    ArrayHandle<std::shared_ptr<AlchemicalParticle>> h_alchemical_particles;
    ArrayHandle<mask_type> h_alchemy_mask;
    };

//! Template class for computing alchemical pair potentials
/*! <b>Overview:</b>

    <b>Implementation details</b>



    \sa export_PotentialPair()
*/
template<class evaluator,
         typename extra_pkg = AlchemyPackage<std::bitset<evaluator::num_alchemical_parameters>>>
class AlchemicalPotentialPair : public PotentialPair<evaluator, extra_pkg>
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
    Index2DUpperTriangular m_alchemy_index; //!< upper triangular typepair index
    GlobalArray<mask_type> m_alchemy_mask;  //!< Type pair mask for if alchemical forces are used
    GlobalArray<std::shared_ptr<AlchemicalParticle>>
        m_alchemical_particles;           //!< 2D array (alchemy_index,alchemical param)
    uint64_t m_next_alchemical_time_step; //!< Next alchemical time step

    //! Method to be called when number of types changes
    virtual void slotNumTypesChange()
        {
        Index2DUpperTriangular new_alchemy_index
            = Index2DUpperTriangular(this->m_pdata->getNTypes());
        GlobalArray<mask_type> new_mask(new_alchemy_index.getNumElements(), this->m_exec_conf);
        GlobalArray<std::shared_ptr<AlchemicalParticle>> new_particles_array(
            new_alchemy_index.getNumElements(),
            evaluator::num_alchemical_parameters,
            this->m_exec_conf);

        ArrayHandle<mask_type> h_new_mask(new_mask, access_location::host, access_mode::overwrite);
        ArrayHandle<std::shared_ptr<AlchemicalParticle>> h_new_particles(new_particles_array,
                                                                         access_location::host,
                                                                         access_mode::overwrite);
        for (unsigned int i = 0; i < new_alchemy_index.getNumElements(); i++)
            {
            h_new_mask.data[i].reset(); // set bitsets to all false by default, enable manually
            for (unsigned int j = 0; j < evaluator::num_alchemical_parameters; j++)
                {
                h_new_particles[j * new_alchemy_index.getNumElements() + i] = nullptr;
                }
            }

        ArrayHandle<std::shared_ptr<AlchemicalParticle>> h_particles(m_alchemical_particles,
                                                                     access_location::host,
                                                                     access_mode::read);
        ArrayHandle<mask_type> h_mask(m_alchemy_mask, access_location::host, access_mode::read);

        // copy over entries that are valid in both the new and old matrices
        unsigned int copy_w = std::min(new_alchemy_index.getW(), m_alchemy_index.getW());
        for (unsigned int i = 0; i < copy_w; i++)
            {
            for (unsigned int j = 0; j < i; j++)
                {
                h_new_mask.data[new_alchemy_index(i, j)] = h_mask.data[m_alchemy_index(i, j)];
                for (unsigned int k = 0; k < evaluator::num_alchemical_parameters; k++)
                    {
                    h_new_particles
                        .data[k * new_alchemy_index.getNumElements() + new_alchemy_index(i, j)]
                        = h_particles
                              .data[k * m_alchemy_index.getNumElements() + m_alchemy_index(i, j)];
                    }
                }
            }
        m_alchemy_index = new_alchemy_index;
        m_alchemical_particles.swap(new_particles_array);
        m_alchemy_mask.swap(new_mask);
        }

    // Extra steps to insert
    inline extra_pkg pkgInitialze(const uint64_t& timestep) override;
    inline void pkgPerNeighbor(const unsigned int& i,
                               const unsigned int& j,
                               const unsigned int& typpair_idx,
                               evaluator& eval,
                               extra_pkg&) override;
    };

template<class evaluator, extra_pkg>
AlchemicalPotentialPair<evaluator, extra_pkg>::AlchemicalPotentialPair(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist,
    const std::string& log_suffix)
    : PotentialPair<evaluator, extra_pkg>(sysdef, nlist, log_suffix)
    {
    this->m_exec_conf->msg->notice(5)
        << "Constructing AlchemicalPotentialPair<" << evaluator::getName() << ">" << std::endl;

    this->m_pdata->getNumTypesChangeSignal()
        .template connect<ALchemicalPotentialPair<evaluator, extra_pkg>,
                          &ALchemicalPotentialPair<evaluator, extra_pkg>::slotNumTypesChange>(this);

    // TODO: subscribe alchemical particles to resize forces?
    // this->m_pdata->getGlobalParticleNumberChangeSignal()
    //     .template connect<
    //         ALchemicalPotentialPair<evaluator, extra_pkg>,
    //         &ALchemicalPotentialPair<evaluator, extra_pkg>::allocateAlchemicalForceArrays>(this);
    }

// TODO: constructor from base class and similar demote for easy switching

template<class evaluator, extra_pkg>
inline extra_pkg
AlchemicalPotentialPair<evaluator, extra_pkg>::pkgInitialze(const uint64_t& timestep)
    {
    // zero force
    if (timestep == m_next_alchemical_time_step)
        {
        this->m_exec_conf->msg->notice(10)
            << "AlchemPotentialPair: Calculating alchemical forces" << std::endl;
        extra_pkg pkg
            = {true,
               ArrayHandle<std::shared_ptr<AlchemicalParticle>>(m_alchemical_particles,
                                                                access_location::host,
                                                                access_mode::read),
               ArrayHandle<mask_type>(m_alchemy_mask, access_location::host, access_mode::read)};
        return pkg;
        }
    else
        {
        extra_pkg pkg;
        pkg.calculate_derivatives = false;
        return pkg;
        }
    // TODO: actually zero forces in each alchemy particle using the memset syntax
    }

template<class evaluator, extra_pkg>
inline void
AlchemicalPotentialPair<evaluator, extra_pkg>::pkgPerNeighbor(const unsigned int& i,
                                                              const unsigned int& j,
                                                              const unsigned int& typpair_idx,
                                                              evaluator& eval,
                                                              extra_pkg& pkg)
    {
    mask_type mask = pkg.h_alchemy_mask.data[m_alchemy_index[i,j];
    // TODO: should we update a copy of the parameters in pkgInitizalize and pass that around
    // instead? yes: more efficient no:need alphas for derivatives anyways currently
    Scalar alphas[evaluator::num_alchemical_parameters];
    for (unsigned int k; k < evaluator::num_alchemical_parameters; k++)
        if ()
            {
            alphas[k] = pkg.h_alchemical_particles
                            .data[k * m_alchemy_index.getNumElements() + m_alchemy_index(i, j)]
                            .get_value()
            }
        else
            {
            alphas[k] = Scalar(1.0)
            }
    // TODO: make sure that when we disable an alchemical particle, we rewrite it's parameter
    eval.AlchemParams(alphas);
    if (pkg.calculate_derivatives)
        {
        Scalar d_alchemical[evaluator::num_alchemical_parameters] = {0.0};
        eval.evalDAlphaEnergy(d_alchemical, alphas);
        }
    }
// per particle neighbor
// // set up alchemical values for the pair (no longer needed)
// for (uint64_t m = 0; m < m_alchem_m; m++)
//     {
//     alpha_ij[m] = Scalar(0.5) * (h_alpha.data[m*m_alchem_pitch+i].x +
//     h_alpha.data[m*m_alchem_pitch+j].x);
//     }
// }

// TODO: This is a literal copy paste, should be possible to improve
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
