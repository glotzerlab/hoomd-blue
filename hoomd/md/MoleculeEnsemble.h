//
// Created by girard01 on 1/20/23.
//

#include "MolecularForceCompute.h"

#ifndef HOOMD_MOLECULEENSEMBLE_H
#define HOOMD_MOLECULEENSEMBLE_H


namespace hoomd::md {
    class MolecularHashAction;
    class MolecularHashCompute;

    class MoleculeEnsemble : public MolecularForceCompute {
    public:
        MoleculeEnsemble(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group,
                         bool include_all_bonded = true) :
        MolecularForceCompute(sysdef), m_group(group){
            if(sysdef->isDomainDecomposed())
                throw std::runtime_error("Molecular ensembles does not work on domain decomposed simulations");
        }

        void initMolecules() override;

        auto& getHashes(){
            return m_hashes;
        }

        void register_action(std::shared_ptr<MolecularHashAction> action);

        void deregister_action(std::shared_ptr<MolecularHashAction> action);

        void computeHashes(std::size_t);

        auto get_molecule_sizes(){
            return m_molecule_size;
        }

        auto get_molecule_indexer(){
            return m_molecule_indexer;
        }

        auto get_molecules(){
            return m_molecules;
        }

    protected:
        std::shared_ptr<ParticleGroup> m_group;

        unsigned int m_hash_size = 0; //! the number of bits currently associated with the hashes of this set of molecules
        GlobalArray<unsigned int> m_hashes; //! the hashes of all molecules within this set
        GlobalArray<Scalar> m_chemical_potentials; //! chemical potentials associated with a given hash, if any

        GlobalArray<unsigned int> m_molecule_size; //! number of beads in each molecule
        GlobalArray<unsigned int> m_molecules; //! indexes of beads of each molecule
        Index2D m_molecule_indexer; //! yields index (molecule, bead_index) into m_molecules

        std::vector<std::weak_ptr<MolecularHashAction>> m_registered_actions;
        std::vector<std::weak_ptr<MolecularHashCompute>> m_registered_computes;
    };
}

#endif //HOOMD_MOLECULEENSEMBLE_H
