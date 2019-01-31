// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "hoomd/ForceConstraint.h"
#include "NeighborList.h"

#ifdef ENABLE_CUDA
#include "hoomd/Autotuner.h"
#endif

/*! \file MolecularForceCompute.h
    \brief Base class for ForceConstraints that depend on a molecular topology

    Implements the data structures that define a molecule topology.
    MolecularForceCompute maintains a list of local molecules and their constituent particles, and
    the particles are sorted according to global particle tag.

    The data structures are initialized by calling initMolecules(). This is done in the derived class
    whenever particles are reordered.

    Every molecule has a unique contiguous tag, 0 <=tag <m_n_molecules_global.

    Derived classes take care of resizing the ghost layer accordingly so that
    spanning molecules are communicated correctly. They connect to the Communicator
    signal using addGhostLayerWidthRequest() .

    In MPI simulations, MolecularForceCompute ensures that local molecules are complete by requesting communication of all
    members of a molecule even if only a single particle member falls into the ghost layer.
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __MolecularForceCompute_H__
#define __MolecularForceCompute_H__

const unsigned int NO_MOLECULE = (unsigned int)0xffffffff;

class PYBIND11_EXPORT MolecularForceCompute : public ForceConstraint
    {
    public:
        //! Constructs the compute
        MolecularForceCompute(std::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~MolecularForceCompute();

        //! Return the number of DOF removed by this constraint
        virtual unsigned int getNDOFRemoved() { return 0; }

        #ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        virtual CommFlags getRequestedCommFlags(unsigned int timestep)
            {
            CommFlags flags = CommFlags(0);

            // request communication of tags
            flags[comm_flag::tag] = 1;

            flags |= ForceConstraint::getRequestedCommFlags(timestep);

            return flags;
            }
        #endif

        //! Return molecule index
        const Index2D& getMoleculeIndexer()
            {
            checkParticlesSorted();

            return m_molecule_indexer;
            }

        //! Return molecule list
        const GlobalVector<unsigned int>& getMoleculeList()
            {
            checkParticlesSorted();

            return m_molecule_list;
            }

        //! Return molecule lengths
        const GlobalVector<unsigned int>& getMoleculeLengths()
            {
            checkParticlesSorted();

            return m_molecule_length;
            }

        //! Return molecule order
        const GlobalVector<unsigned int>& getMoleculeOrder()
            {
            checkParticlesSorted();

            return m_molecule_order;
            }

        //! Return reverse lookup array
        const GlobalVector<unsigned int>& getMoleculeIndex()
            {
            checkParticlesSorted();

            return m_molecule_idx;
            }

        #ifdef ENABLE_CUDA
        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs

            Derived classes should override this to set the parameters of their autotuners.
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            if (m_exec_conf->isCUDAEnabled())
                {
                m_tuner_fill->setPeriod(period);
                m_tuner_fill->setEnabled(enable);
                }
            }
        #endif

    protected:
        GlobalVector<unsigned int> m_molecule_tag;     //!< Molecule tag per particle tag
        unsigned int m_n_molecules_global;          //!< Global number of molecules

        bool m_dirty;                               //!< True if we need to rebuild indices

        //! Helper function to check if particles have been sorted and rebuild indices if necessary
        virtual void checkParticlesSorted()
            {
            if (m_dirty)
                {
                // rebuild molecule list
                initMolecules();
                m_dirty = false;
                }
            }

    private:
        GlobalVector<unsigned int> m_molecule_list;    //!< 2D Array of molecule members
        GlobalVector<unsigned int> m_molecule_length;  //!< List of molecule lengths
        GlobalVector<unsigned int> m_molecule_order;   //!< Order in molecule by local ptl idx
        GlobalVector<unsigned int> m_molecule_idx;     //!< Reverse-lookup into molecule list

        #ifdef ENABLE_CUDA
        std::unique_ptr<Autotuner> m_tuner_fill;    //!< Autotuner for block size for filling the molecule table
        #endif

        Index2D m_molecule_indexer;                 //!< Index of the molecule table

        void setDirty()
            {
            m_dirty = true;
            }

        //! construct a list of local molecules
        virtual void initMolecules();

        #ifdef ENABLE_CUDA
        //! construct a list of local molecules on the GPU
        virtual void initMoleculesGPU();
        #endif

        #ifdef ENABLE_CUDA
        GPUPartition m_gpu_partition;               //!< Partition of the molecules on GPUs
        #endif
    };

//! Exports the MolecularForceCompute to python
void export_MolecularForceCompute(pybind11::module& m);

#endif
