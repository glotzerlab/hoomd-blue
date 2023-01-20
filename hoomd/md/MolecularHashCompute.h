//
// Created by girard01 on 1/20/23.
//

#include "hoomd/Compute.h"
#include <hoomd/Updater.h>
#include "MoleculeEnsemble.h"

#ifndef HOOMD_MOLECULARHASHCOMPUTE_H
#define HOOMD_MOLECULARHASHCOMPUTE_H

namespace hoomd::md{

    class MolecularHashAction{
        friend class MoleculeEnsemble;
    public:
        MolecularHashAction(std::shared_ptr<MoleculeEnsemble> ensemble):
        m_target_ensemble(ensemble){}

        virtual unsigned char get_required_bits(){
            return 0;
        }

        virtual void initialize(){} //! initializes hash values, this needs to create a valid state for the hashes indepedently of its current state

    protected:
        unsigned int hash_offset = 0;
        std::shared_ptr<MoleculeEnsemble> m_target_ensemble;
    };

    class MolecularHashCompute : public MolecularHashAction, Compute{
    public:
        MolecularHashCompute(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<MoleculeEnsemble> ensemble) :
        Compute(sysdef), MolecularHashAction(ensemble){}


        void compute(std::size_t timestep) override{} //! computes the current hashes of all molecules in the target_ensemble

        void initialize() override{  //! initialize the hashes by computing this for t = 0
            this->forceCompute(0);
        }
    };

    class MolecularHashUpdater : public MolecularHashAction, Updater{
    public:
        MolecularHashUpdater(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<Trigger> trigger,
                             std::shared_ptr<MoleculeEnsemble> ensemble):
                Updater(sysdef, trigger), MolecularHashAction(ensemble){}

        void update(std::size_t) override{}; //! updates the current hash of all molecules
    };
}

#endif //HOOMD_MOLECULARHASHCOMPUTE_H
