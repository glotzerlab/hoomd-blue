// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "NeighborList.h"
#include "hoomd/ForceConstraint.h"

#ifdef ENABLE_HIP
#include "hoomd/Autotuner.h"
#endif

/*! \file MolecularForceCompute.h
    \brief Base class for ForceConstraints that depend on a molecular topology

    Implements the data structures that define a molecule topology.
    MolecularForceCompute maintains a list of local molecules and their constituent particles, and
    the particles are sorted according to global particle tag.

    The data structures are initialized by calling initMolecules(). This is done in the derived
   class whenever particles are reordered.

    Every molecule has a unique contiguous tag, 0 <=tag <m_n_molecules_global.

    Derived classes take care of resizing the ghost layer accordingly so that
    spanning molecules are communicated correctly. They connect to the Communicator
    signal using addGhostLayerWidthRequest() .

    In MPI simulations, MolecularForceCompute ensures that local molecules are complete by
   requesting communication of all members of a molecule even if only a single particle member falls
   into the ghost layer.
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __MolecularForceCompute_H__
#define __MolecularForceCompute_H__

const unsigned int NO_MOLECULE = (unsigned int)0xffffffff;

namespace hoomd
    {
namespace md
    {
class PYBIND11_EXPORT MolecularForceCompute : public ForceConstraint
    {
    public:
    //! Constructs the compute
    MolecularForceCompute(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~MolecularForceCompute();

    //! Return the number of DOF removed by this constraint
    virtual Scalar getNDOFRemoved(std::shared_ptr<ParticleGroup> query)
        {
        return 0;
        }

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
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

    /// Get the number of molecules (global)
    unsigned int getNMoleculesGlobal() const
        {
        return m_n_molecules_global;
        }

    protected:
    GlobalVector<unsigned int> m_molecule_tag; //!< Molecule tag per particle tag
    unsigned int m_n_molecules_global;         //!< Global number of molecules

    bool m_rebuild_molecules; //!< True if we need to rebuild indices

    //! Helper function to check if particles have been sorted and rebuild indices if necessary
    virtual void checkParticlesSorted()
        {
        if (m_rebuild_molecules)
            {
            // rebuild molecule list
            initMolecules();
            m_rebuild_molecules = false;
            }
        }

    private:
    /// 2D Array of molecule members. Use m_molecule_indexer to index into this array. The data
    /// stored is
    /// m_molecule_list[
    ///     m_molecule_indexer(particle_molecule_index, molecule_index)
    /// ] == local_particle_index
    GlobalVector<unsigned int> m_molecule_list;

    /// List of molecule lengths
    GlobalVector<unsigned int> m_molecule_length;

    /// Index of particle in a molecule. Accessed through local particle index.
    GlobalVector<unsigned int> m_molecule_order;

    /// Reverse-lookup into molecule list, specifically
    /// m_molecule_idx[particle_index] == / molecule_index (note that this is the temporary
    /// particle index not the permanent particle tag).
    GlobalVector<unsigned int> m_molecule_idx;

#ifdef ENABLE_HIP
    std::shared_ptr<Autotuner<1>>
        m_tuner_fill; //!< Autotuner for block size for filling the molecule table
#endif

    /// Functor for indexing into a 1D array as if it were a 2-D array. Index is
    /// [constituent_number, molecule_number].
    Index2D m_molecule_indexer;

    void setRebuildMolecules()
        {
        m_rebuild_molecules = true;
        }

    //! construct a list of local molecules
    virtual void initMolecules();

#ifdef ENABLE_HIP
    //! construct a list of local molecules on the GPU
    virtual void initMoleculesGPU();
#endif

#ifdef ENABLE_HIP
    GPUPartition m_gpu_partition; //!< Partition of the molecules on GPUs
#endif
    };

    } // end namespace md
    } // end namespace hoomd

#endif
