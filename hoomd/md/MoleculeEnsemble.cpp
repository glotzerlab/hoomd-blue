//
// Created by girard01 on 1/20/23.
//

#include "MoleculeEnsemble.h"
#include "MolecularHashCompute.h"

namespace hoomd::md{

    void MoleculeEnsemble::register_action(
            std::shared_ptr<MolecularHashAction> action) {
        auto required_bits = action->get_required_bits();

        if(m_hash_size + required_bits > 32)
            throw std::runtime_error("Requested hash size would require more than 2^32 types of molecules!");

        action->hash_offset = m_hash_size;
        m_hash_size += required_bits;
        m_registered_actions.push_back(action);

        auto compute_pointer = std::dynamic_pointer_cast<MolecularHashCompute>(action);
        if(compute_pointer)
            m_registered_computes.push_back(compute_pointer);
        action->initialize();
    }

    void MoleculeEnsemble::computeHashes(std::size_t timestep) {
        for(auto&& compute : m_registered_computes) {
            auto _compute = compute.lock();
            if(_compute)
                _compute->compute(timestep);
        }
    }

    void MoleculeEnsemble::deregister_action(
            std::shared_ptr<MolecularHashAction> action) {
        auto pos_actions =std::find_if(m_registered_actions.begin(),
                                       m_registered_actions.end(), [&action](auto& element){return action == element.lock();});
        if(pos_actions == m_registered_actions.end())
            throw std::runtime_error("Trying to deregister an unregistered action");

        auto removed_action_hash_offset = action->hash_offset;
        auto removed_action_hash_size = action->get_required_bits();
        m_registered_actions.erase(pos_actions);

        // we need to adjust all hash sizes and reinitialize every associated hash compute
        for(auto&& registerd_action : m_registered_actions){
            auto _action = registerd_action.lock();
            if(_action && _action->hash_offset > removed_action_hash_offset) {
                _action->hash_offset -= removed_action_hash_size;
                _action->initialize();
            }
        }

        // check if its in the registered computes and remove it
        auto pos = std::find_if(m_registered_computes.begin(),
                                 m_registered_computes.end(), [&action](auto& element){return action == element.lock();});
        if(pos != m_registered_computes.end())
            m_registered_computes.erase(pos);
    }

    namespace detail{

        void export_MoleculeEnsemble(pybind11::module& m){
            pybind11::class_<MoleculeEnsemble, MolecularForceCompute, std::shared_ptr<MoleculeEnsemble>>(m, "MoleculeEnsemble")
            .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, bool>());
        }

    }
}