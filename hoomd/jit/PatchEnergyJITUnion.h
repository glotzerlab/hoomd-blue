#ifndef _PATCH_ENERGY_JIT_UNION_H_
#define _PATCH_ENERGY_JIT_UNION_H_

#include "PatchEnergyJIT.h"
#include "hoomd/hpmc/GPUTree.h"
#include "hoomd/SystemDefinition.h"
#include "hoomd/managed_allocator.h"

//! Evaluate patch energies via runtime generated code, using a tree accelerator structure for unions of particles
class PatchEnergyJITUnion : public PatchEnergyJIT
    {
    public:
        //! Constructor
        /*! \param r_cut Max rcut for constituent particles
         */
        PatchEnergyJITUnion(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ExecutionConfiguration> exec_conf,
            const std::string& llvm_ir_iso, Scalar r_cut_iso,
            const unsigned int array_size_iso,
            const std::string& llvm_ir_union, Scalar r_cut_union,
            const unsigned int array_size_union)
            : PatchEnergyJIT(exec_conf, llvm_ir_iso, r_cut_iso, array_size_iso), m_sysdef(sysdef),
            m_rcut_union(r_cut_union),
            m_alpha_union(array_size_union, 0.0f, managed_allocator<float>(m_exec_conf->isCUDAEnabled())),
            m_alpha_size_union(array_size_union)
            {
            // build the JIT.
            m_factory_union = std::shared_ptr<EvalFactory>(new EvalFactory(llvm_ir_union));

            // get the evaluator
            m_eval_union = m_factory_union->getEval();

            if (!m_eval_union)
                {
                exec_conf->msg->error() << m_factory_union->getError() << std::endl;
                throw std::runtime_error("Error compiling Union JIT code.");
                }

            m_factory_union->setAlphaUnionArray(&m_alpha_union.front());

            // Connect to number of types change signal
            m_sysdef->getParticleData()->getNumTypesChangeSignal().connect<PatchEnergyJITUnion, &PatchEnergyJITUnion::slotNumTypesChange>(this);

            unsigned int ntypes = m_sysdef->getParticleData()->getNTypes();
            m_extent_type.resize(ntypes,0.0);
            m_type.resize(ntypes);
            m_position.resize(ntypes);
            m_orientation.resize(ntypes);
            m_diameter.resize(ntypes);
            m_charge.resize(ntypes);
            m_tree.resize(ntypes);
            }

        //! Destructor
        virtual ~PatchEnergyJITUnion()
            {
            m_sysdef->getParticleData()->getNumTypesChangeSignal().disconnect<PatchEnergyJITUnion, &PatchEnergyJITUnion::slotNumTypesChange>(this);
            }

        //! Set the per-type constituent particles
        /*! \param type The particle type to set the constituent particles for
            \param rcut The maximum cutoff over all constituent particles for this type
            \param types The type IDs for every constituent particle
            \param positions The positions
            \param orientations The orientations
            \param leaf_capacity Number of particles in OBB tree leaf
         */
        virtual void setParam(unsigned int type,
                              std::vector<unsigned int> types,
                              std::vector<vec3<float> > positions,
                              std::vector<quat<float> > orientations,
                              std::vector<float> diameters,
                              std::vector<float> charges,
                              unsigned int leaf_capacity);

        virtual void setTypeID(std::string type, pybind11::list typeids)
            {
            unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
            std::vector<unsigned int> typeid_vec(pybind11::len(typeids));
            for (unsigned int i = 0; i < typeid_vec.size(); i++)
                {
                typeid_vec[i] = pybind11::cast<unsigned int>(typeids[i]);
                }
            setParam(pid, typeid_vec, m_position[pid], m_orientation[pid], m_diameter[pid], m_charge[pid], m_leaf_capacity);
            }

        virtual pybind11::list getTypeID(std::string type)
            {
            unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
            pybind11::list ret;
            for (unsigned int i = 0; i < m_type[pid].size(); i++)
                {
                ret.append(m_type[pid][i]);
                }
            return ret;
            }

        virtual void setPositions(std::string type, pybind11::list position)
            {
            unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
            std::vector<vec3<float> > pos_vec(pybind11::len(position));
            for (unsigned int i = 0; i < pos_vec.size(); i++)
                {
                pybind11::list p_i = position[i];
                vec3<float> pos(p_i[0].cast<float>(),
                                p_i[1].cast<float>(),
                                p_i[2].cast<float>());
                pos_vec[i] = pos;
                }
            setParam(pid, m_type[pid], pos_vec, m_orientation[pid], m_diameter[pid], m_charge[pid], m_leaf_capacity);
            }

        virtual pybind11::list getPositions(std::string type)
            {
            unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
            pybind11::list ret;
            for (unsigned int i = 0; i < m_position[pid].size(); i++)
                {
                pybind11::list tmp;
                tmp.append(m_position[pid][i].x);
                tmp.append(m_position[pid][i].y);
                tmp.append(m_position[pid][i].z);
                ret.append(tmp);
                }
            return ret;
            }

        virtual void setOrientations(std::string type, pybind11::list orientation)
            {
            unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
            std::vector<quat<float>> ort_vec(pybind11::len(orientation));
            for (unsigned int i = 0; i < ort_vec.size(); i++)
                {
                pybind11::list q_i = orientation[i];
                float s = q_i[0].cast<float>();
                float x = q_i[1].cast<float>();
                float y = q_i[2].cast<float>();
                float z = q_i[3].cast<float>();
                quat<float> ort(s, vec3<float>(x,y,z));
                ort_vec[i] = ort;
                }
            setParam(pid, m_type[pid], m_position[pid], ort_vec, m_diameter[pid], m_charge[pid], m_leaf_capacity);
            }

        virtual pybind11::list getOrientations(std::string type)
            {
            unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
            pybind11::list ret;
            for (unsigned int i = 0; i < m_orientation[pid].size(); i++)
                {
                pybind11::list tmp;
                tmp.append(m_orientation[pid][i].s);
                tmp.append(m_orientation[pid][i].v.x);
                tmp.append(m_orientation[pid][i].v.y);
                tmp.append(m_orientation[pid][i].v.z);
                ret.append(tmp);
                }
            return ret;
            }

        virtual void setDiameters(std::string type, pybind11::list diameter)
            {
            unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
            std::vector<float> diameter_vec(pybind11::len(diameter));
            for (unsigned int i = 0; i < diameter_vec.size(); i++)
                {
                diameter_vec[i] = diameter[i].cast<float>();
                }
            setParam(pid, m_type[pid], m_position[pid], m_orientation[pid], diameter_vec, m_charge[pid], m_leaf_capacity);
            }

        virtual pybind11::list getDiameters(std::string type)
            {
            unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
            pybind11::list ret;
            for (unsigned int i = 0; i < m_diameter[pid].size(); i++)
                {
                ret.append(m_diameter[pid][i]);
                }
            return ret;
            }

        virtual void setCharges(std::string type, pybind11::list charge)
            {
            unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
            std::vector<float> charge_vec(pybind11::len(charge));
            for (unsigned int i=0; i<charge_vec.size(); i++)
                {
                charge_vec[i] = charge[i].cast<float>();
                }
            setParam(pid, m_type[pid], m_position[pid], m_orientation[pid], m_diameter[pid], charge_vec, m_leaf_capacity);
            }

        virtual pybind11::list getCharges(std::string type)
            {
            unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
            pybind11::list ret;
            for (unsigned int i = 0; i < m_charge[pid].size(); i++)
                {
                ret.append(m_charge[pid][i]);
                }
            return ret;
            }

        virtual void setLeafCapacity(unsigned int leaf_capacity)
            {
            setParam(0, m_type[0], m_position[0], m_orientation[0], m_diameter[0], m_charge[0], leaf_capacity);
            }

        virtual unsigned int getLeafCapacity()
            {
            return m_leaf_capacity;
            }

        //! Get the cut-off for constituent particles
        virtual Scalar getRCut()
            {
            // return cutoff for constituent particle potentials
            return m_rcut_union;
            }

        //! Get the maximum geometric extent, which is added to the cutoff, per type
        virtual inline Scalar getAdditiveCutoff(unsigned int type)
            {
            assert(type <= m_extent_type.size());
            Scalar extent = m_extent_type[type];
            // ensure the minimum cutoff distance is the isotropic r_cut
            if (extent + m_rcut_union < m_r_cut)
                return m_r_cut-m_rcut_union;
            else
                return extent;
            }

        //! evaluate the energy of the patch interaction
        /*! \param r_ij Vector pointing from particle i to j
            \param type_i Integer type index of particle i
            \param d_i Diameter of particle i
            \param charge_i Charge of particle i
            \param q_i Orientation quaternion of particle i
            \param type_j Integer type index of particle j
            \param q_j Orientation quaternion of particle j
            \param d_j Diameter of particle j
            \param charge_j Charge of particle j

            \returns Energy of the patch interaction.
        */
        virtual float energy(const vec3<float>& r_ij,
            unsigned int type_i,
            const quat<float>& q_i,
            float diameter_i,
            float charge_i,
            unsigned int type_j,
            const quat<float>& q_j,
            float d_j,
            float charge_j);

        //! Method to be called when number of types changes
        virtual void slotNumTypesChange()
            {
            unsigned int ntypes = m_sysdef->getParticleData()->getNTypes();
            m_extent_type.resize(ntypes,0.0);
            m_type.resize(ntypes);
            m_position.resize(ntypes);
            m_orientation.resize(ntypes);
            m_diameter.resize(ntypes);
            m_charge.resize(ntypes);
            m_tree.resize(ntypes);
            }

        static pybind11::object getAlphaUnionNP(pybind11::object self)
            {
            auto self_cpp = self.cast<PatchEnergyJITUnion *>();
            return pybind11::array(self_cpp->m_alpha_size_union, self_cpp->m_factory_union->getAlphaUnionArray(), self);
            }

    protected:
        std::shared_ptr<SystemDefinition> m_sysdef;               // HOOMD's system definition
        std::vector<hpmc::detail::GPUTree> m_tree;                // The tree acceleration structure per particle type
        std::vector< float > m_extent_type;                       // The per-type geometric extent
        std::vector< std::vector<vec3<float> > > m_position;      // The positions of the constituent particles
        std::vector< std::vector<quat<float> > > m_orientation;   // The orientations of the constituent particles
        std::vector< std::vector<float> > m_diameter;             // The diameters of the constituent particles
        std::vector< std::vector<float> > m_charge;               // The charges of the constituent particles
        std::vector< std::vector<unsigned int> > m_type;          // The type identifiers of the constituent particles
        unsigned int m_leaf_capacity;                             // The number of particles in a leaf of the internal tree data structure

        //! Compute the energy of two overlapping leaf nodes
        float compute_leaf_leaf_energy(vec3<float> dr,
                                     unsigned int type_a,
                                     unsigned int type_b,
                                     const quat<float>& orientation_a,
                                     const quat<float>& orientation_b,
                                     unsigned int cur_node_a,
                                     unsigned int cur_node_b);

        std::shared_ptr<EvalFactory> m_factory_union;            //!< The factory for the evaluator function, for constituent ptls
        EvalFactory::EvalFnPtr m_eval_union;                     //!< Pointer to evaluator function inside the JIT module
        Scalar m_rcut_union;                                     //!< Cutoff on constituent particles
        std::vector<float, managed_allocator<float> > m_alpha_union; //!< Data array for union
        unsigned int m_alpha_size_union;
    };

//! Exports the PatchEnergyJITUnion class to python
void export_PatchEnergyJITUnion(pybind11::module &m);
#endif // _PATCH_ENERGY_JIT_UNION_H_
