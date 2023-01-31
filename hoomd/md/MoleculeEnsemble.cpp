//
// Created by girard01 on 1/20/23.
//

#include "MoleculeEnsemble.h"
#include "MolecularHashCompute.h"

void updateMoleculeList(std::vector<std::vector<unsigned int>>& molecules,
                        std::vector<unsigned int>& tagLookup,
                        const std::pair<unsigned int, unsigned int> newPair){
    // this takes a new pair of two bonded components, and updates molecule list and molecule lookups
    auto idx_i = newPair.first;
    auto idx_j = newPair.second;

    if(tagLookup[idx_i] == NO_MOLECULE && tagLookup[idx_j] == NO_MOLECULE){
        auto molecule_idx = static_cast<unsigned int>(molecules.size());
        std::vector<unsigned int> molecule = {idx_i, idx_j};
        molecules.push_back(molecule);
        tagLookup[idx_i] = tagLookup[idx_j] = molecule_idx;
    }
    else if(tagLookup[idx_i] == NO_MOLECULE){
        molecules[tagLookup[idx_j]].push_back(idx_i);
        tagLookup[idx_i] = tagLookup[idx_j];
    }
    else if(tagLookup[idx_j] == NO_MOLECULE){
        molecules[tagLookup[idx_i]].push_back(idx_j);
        tagLookup[idx_j] = tagLookup[idx_i];
    }
    else if (tagLookup[idx_i] != tagLookup[idx_j]){
        if(tagLookup[idx_i] < tagLookup[idx_j])
            std::swap(idx_i, idx_j);
        auto molecule_i = tagLookup[idx_i];
        auto molecule_j = tagLookup[idx_j];
        for(auto&& tag_j : molecules[molecule_j]) {
            molecules[molecule_i].push_back(tag_j);
            tagLookup[tag_j] = molecule_i;
        }
        molecules.erase(molecules.begin() + molecule_j);
    }
}


namespace hoomd::md{

    MoleculeEnsemble::MoleculeEnsemble(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<ParticleGroup> group,
                                       bool include_all_bonded) :
                                       MolecularForceCompute(sysdef), m_group(group), m_include_all_bonded(include_all_bonded){
        if(sysdef->isDomainDecomposed())
            throw std::runtime_error("Molecular ensembles does not work on domain decomposed simulations");
        rebuild_table();
    }

    void MoleculeEnsemble::rebuild_table() {

        std::vector<std::vector<unsigned int>> molecules;
        std::vector<unsigned int> tag_lookup(m_pdata->getNGlobal(), NO_MOLECULE);

        ArrayHandle<unsigned int> h_r_tags(m_pdata->getRTags(), access_location::host, access_mode::read);

        const auto& bond_data = m_sysdef->getBondData();
        for(auto i = 0; i < bond_data->getNGlobal(); i++){
            auto& bond_tags = bond_data->getMembersByIndex(i).tag;
            auto within_group = std::make_pair(m_group->isMember(h_r_tags.data[bond_tags[0]]),
                                  m_group->isMember(h_r_tags.data[bond_tags[1]]));

            auto condition = (within_group.first && within_group.second) || m_include_all_bonded;
            updateMoleculeList(molecules, tag_lookup, {bond_tags[0], bond_tags[1]});
        }

        // list may contain molecule with no beads from m_group if m_include_all_bonded is on
        for(auto it = molecules.begin(); it != molecules.end();){
            auto mol = *it;
            auto is_within_group = std::accumulate(mol.begin(), mol.end(), false, [this, &h_r_tags](bool sum, auto element){return sum | m_group->isMember(h_r_tags.data[element]);});
            if(!is_within_group){ // remove this molecule from the list
                for(auto idx : mol)
                    tag_lookup[idx] = NO_MOLECULE;
                molecules.erase(it);
            }else{
                it++;
            }
        }

        // rebuild the tag lookup vector
        for(auto i = 0; i < molecules.size(); i++){
            auto mol = molecules[i];
            for(const auto& el : mol)
                tag_lookup[el] = i;
        }
        m_n_molecules_global = molecules.size();
        ArrayHandle<unsigned int> h_molecule_tag(m_molecule_tag, access_location::host, access_mode::overwrite);
        std::copy(tag_lookup.begin(), tag_lookup.end(), h_molecule_tag.data);
    }

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

    auto MoleculeEnsemble::get_hash_description(unsigned int hash) {
        std::vector<std::string> description;
        unsigned char offset = 0;
        for(auto& _compute : m_registered_computes){
            auto compute = _compute.lock();
            auto _hash = hash >> offset;
            auto mask = (1u << compute->get_required_bits()) - 1;
            auto compute_description = compute->get_description(_hash & mask);
            offset += compute->get_required_bits();
            if(compute_description.empty())
                return std::vector<std::string>();
            description.push_back(compute_description);
        }
        return description;
    }

    namespace detail{

        void export_MoleculeEnsemble(pybind11::module& m){
            pybind11::class_<MoleculeEnsemble, MolecularForceCompute, std::shared_ptr<MoleculeEnsemble>>(m, "MoleculeEnsemble")
            .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, bool>())
            .def("getHashSize", &MoleculeEnsemble::getMaximumHash)
            .def("getHashDescription", &MoleculeEnsemble::get_hash_description)
            .def("registerAction", &MoleculeEnsemble::register_action)
            .def("deregisterAction", &MoleculeEnsemble::deregister_action);
        }

    }
}