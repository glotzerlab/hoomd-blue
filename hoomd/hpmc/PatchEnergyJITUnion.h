#ifndef _PATCH_ENERGY_JIT_UNION_H_
#define _PATCH_ENERGY_JIT_UNION_H_

#include "PatchEnergyJIT.h"
#include "hoomd/SystemDefinition.h"
#include "hoomd/hpmc/GPUTree.h"
#include "hoomd/managed_allocator.h"

//! Evaluate patch energies via runtime generated code, using a tree accelerator structure for
//! unions of particles
class PatchEnergyJITUnion : public PatchEnergyJIT
    {
    public:
    //! Constructor
    /*! \param r_cut Max rcut for constituent particles
     */
    PatchEnergyJITUnion(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<ExecutionConfiguration> exec_conf,
                        const std::string& cpp_code_iso,
                        const std::vector<std::string>& compiler_args,
                        Scalar r_cut_iso,
                        pybind11::array_t<float> param_array,
                        const std::string& cpu_code_union,
                        Scalar r_cut_constituent,
                        const unsigned int array_size_union)
        : PatchEnergyJIT(sysdef, exec_conf, cpp_code_iso, compiler_args, r_cut_iso, param_array), m_sysdef(sysdef),
          m_r_cut_constituent(r_cut_constituent),
          m_alpha_union(array_size_union,
                        0.0f,
                        managed_allocator<float>(m_exec_conf->isCUDAEnabled())),
          m_alpha_size_union(array_size_union)
        {
        // build the JIT.
        m_factory_union = std::shared_ptr<EvalFactory>(new EvalFactory(cpu_code_union, compiler_args));

        // get the evaluator
        m_eval_union = m_factory_union->getEval();

        if (!m_eval_union)
            {
            std::ostringstream s;
            s << "Error compiling JIT code:" << std::endl;
            s << cpu_code_union << std::endl;
            s << m_factory->getError() << std::endl;
            throw std::runtime_error(s.str());
            }

        m_factory_union->setAlphaUnionArray(&m_alpha_union.front());

        unsigned int ntypes = m_sysdef->getParticleData()->getNTypes();
        m_extent_type.resize(ntypes, 0.0);
        m_type.resize(ntypes);
        m_position.resize(ntypes);
        m_orientation.resize(ntypes);
        m_diameter.resize(ntypes);
        m_charge.resize(ntypes);
        m_tree.resize(ntypes);

        m_managed_memory = false;
        }

    //! Destructor
    virtual ~PatchEnergyJITUnion() { }

    // //! Builds OBB tree based on geometric properties of the constituent particles
    // //! and the leaf capacity. To be called every time positions, diameters and/or leaf
    // //! leaf capacity are updated.
    void buildOBBTree();

    //! Set per-type typeid of constituent particles
    virtual void setTypeids(std::string type, pybind11::list typeids)
        {
        unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
        unsigned int N = (unsigned int)pybind11::len(typeids);
        m_type[pid].resize(N);
        for (unsigned int i = 0; i < N; i++)
            {
            m_type[pid][i] = pybind11::cast<unsigned int>(typeids[i]);
            }
        }

    //! Get per-type typeid of constituent particles as a python list
    virtual pybind11::list getTypeids(std::string type)
        {
        unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
        pybind11::list ret;
        for (unsigned int i = 0; i < m_type[pid].size(); i++)
            {
            ret.append(m_type[pid][i]);
            }
        return ret;
        }

    //! Set per-type positions of the constituent particles
    virtual void setPositions(std::string type, pybind11::list position)
        {
        unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
        unsigned int N = (unsigned int)pybind11::len(position);
        m_position[pid].resize(N);
        for (unsigned int i = 0; i < N; i++)
            {
            pybind11::tuple p_i = position[i];
            vec3<float> pos(p_i[0].cast<float>(), p_i[1].cast<float>(), p_i[2].cast<float>());
            m_position[pid][i] = pos;
            }
        if (std::find(m_updated_types.begin(), m_updated_types.end(), pid) == m_updated_types.end())
            {
            m_updated_types.push_back(pid);
            }
        m_build_obb = true;
        }

    //! Get per-type positions of the constituent particles as a python list of 3-tuples
    virtual pybind11::list getPositions(std::string type)
        {
        unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
        pybind11::list ret;
        for (unsigned int i = 0; i < m_position[pid].size(); i++)
            {
            pybind11::tuple tmp = pybind11::make_tuple(m_position[pid][i].x,
                                                       m_position[pid][i].y,
                                                       m_position[pid][i].z);
            ret.append(tmp);
            }
        return ret;
        }

    //! Set per-type positions of the constituent particles
    virtual void setOrientations(std::string type, pybind11::list orientation)
        {
        unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
        unsigned int N = (unsigned int)pybind11::len(orientation);
        m_orientation[pid].resize(N);
        for (unsigned int i = 0; i < N; i++)
            {
            pybind11::tuple q_i = orientation[i];
            float s = q_i[0].cast<float>();
            float x = q_i[1].cast<float>();
            float y = q_i[2].cast<float>();
            float z = q_i[3].cast<float>();
            quat<float> ort(s, vec3<float>(x, y, z));
            m_orientation[pid][i] = ort;
            }
        }

    //! Get per-type orientations of the constituent particles as a python list of 4-tuples
    virtual pybind11::list getOrientations(std::string type)
        {
        unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
        pybind11::list ret;
        for (unsigned int i = 0; i < m_orientation[pid].size(); i++)
            {
            pybind11::tuple tmp = pybind11::make_tuple(m_orientation[pid][i].s,
                                                       m_orientation[pid][i].v.x,
                                                       m_orientation[pid][i].v.y,
                                                       m_orientation[pid][i].v.z);
            ret.append(tmp);
            }
        return ret;
        }

    //! Set per-type diameters of the constituent particles
    virtual void setDiameters(std::string type, pybind11::list diameter)
        {
        unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
        unsigned int N = (unsigned int)pybind11::len(diameter);
        m_diameter[pid].resize(N);
        for (unsigned int i = 0; i < N; i++)
            {
            m_diameter[pid][i] = diameter[i].cast<float>();
            }
        if (std::find(m_updated_types.begin(), m_updated_types.end(), pid) == m_updated_types.end())
            {
            m_updated_types.push_back(pid);
            }
        m_build_obb = true;
        }

    //! Get per-type diameters of the constituent particles as a python list
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

    //! Set per-type charges of the constituent particles
    virtual void setCharges(std::string type, pybind11::list charge)
        {
        unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
        unsigned int N = (unsigned int)pybind11::len(charge);
        m_charge[pid].resize(N);
        for (unsigned int i = 0; i < N; i++)
            {
            m_charge[pid][i] = charge[i].cast<float>();
            }
        }

    //! Get per-type charges of the constituent particles as python list
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

    //! Set OBB leaf_capacity
    virtual void setLeafCapacity(unsigned int leaf_capacity)
        {
        m_leaf_capacity = leaf_capacity;
        m_build_obb = true;
        }

    //! Get OBB leaf_capacity
    virtual unsigned int getLeafCapacity()
        {
        return m_leaf_capacity;
        }

    //! Get the maximum r_ij radius beyond which energies are always 0
    virtual Scalar getRCut()
        {
        return m_r_cut_constituent;
        }

    //! Get the cut-off for constituent particles
    virtual Scalar getRCutConstituent()
        {
        // return cutoff for constituent particle potentials
        return m_r_cut_constituent;
        }

    //! Set the cut-off for constituent particles
    virtual void setRCutConstituent(Scalar r_cut)
        {
        // return cutoff for constituent particle potentials
        m_r_cut_constituent = r_cut;
        }

    //! Get the size of the alpha_union array
    unsigned int getArraySizeUnion()
        {
        return m_alpha_size_union;
        }

    //! Get the maximum geometric extent, which is added to the cutoff, per type
    virtual inline Scalar getAdditiveCutoff(unsigned int type)
        {
        assert(type <= m_extent_type.size());
        Scalar extent = m_extent_type[type];
        // ensure the minimum cutoff distance is the isotropic r_cut
        if (0.5*extent + m_r_cut_constituent < m_r_cut_isotropic)
            return m_r_cut_isotropic - m_r_cut_constituent;
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
        m_extent_type.resize(ntypes, 0.0);
        m_type.resize(ntypes);
        m_position.resize(ntypes);
        m_orientation.resize(ntypes);
        m_diameter.resize(ntypes);
        m_charge.resize(ntypes);
        m_tree.resize(ntypes);
        }

    static pybind11::object getAlphaUnionNP(pybind11::object self)
        {
        auto self_cpp = self.cast<PatchEnergyJITUnion*>();
        return pybind11::array(self_cpp->m_alpha_size_union,
                               self_cpp->m_factory_union->getAlphaUnionArray(),
                               self);
        }

    protected:
    std::shared_ptr<SystemDefinition> m_sysdef; // HOOMD's system definition
    std::vector<hpmc::detail::GPUTree> m_tree;  // The tree acceleration structure per particle type
    std::vector<float> m_extent_type;           // The per-type geometric extent
    std::vector<std::vector<vec3<float>>> m_position; // The positions of the constituent particles
    std::vector<std::vector<quat<float>>>
        m_orientation;                          // The orientations of the constituent particles
    std::vector<std::vector<float>> m_diameter; // The diameters of the constituent particles
    std::vector<std::vector<float>> m_charge;   // The charges of the constituent particles
    std::vector<std::vector<unsigned int>>
        m_type; // The type identifiers of the constituent particles
    unsigned int
        m_leaf_capacity;   // The number of particles in a leaf of the internal tree data structure
    bool m_managed_memory; // Flag to managed memory on the GPU (only used for OBB construction)

    //! Compute the energy of two overlapping leaf nodes
    float compute_leaf_leaf_energy(vec3<float> dr,
                                   unsigned int type_a,
                                   unsigned int type_b,
                                   const quat<float>& orientation_a,
                                   const quat<float>& orientation_b,
                                   unsigned int cur_node_a,
                                   unsigned int cur_node_b);

    std::shared_ptr<EvalFactory>
        m_factory_union; //!< The factory for the evaluator function, for constituent ptls
    EvalFactory::EvalFnPtr m_eval_union; //!< Pointer to evaluator function inside the JIT module
    Scalar m_r_cut_constituent;                 //!< Cutoff on constituent particles
    std::vector<float, managed_allocator<float>> m_alpha_union; //!< Data array for union
    unsigned int m_alpha_size_union;                            //!< Size of the alpha_union array
    std::vector<unsigned int>
        m_updated_types; //!< List of types whose geometric properties were updated
    };

//! Exports the PatchEnergyJITUnion class to python
void export_PatchEnergyJITUnion(pybind11::module& m);
#endif // _PATCH_ENERGY_JIT_UNION_H_
