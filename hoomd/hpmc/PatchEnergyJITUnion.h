// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _PATCH_ENERGY_JIT_UNION_H_
#define _PATCH_ENERGY_JIT_UNION_H_

#include "PatchEnergyJIT.h"
#include "hoomd/SystemDefinition.h"
#include "hoomd/hpmc/GPUTree.h"
#include "hoomd/managed_allocator.h"

namespace hoomd
    {
namespace hpmc
    {
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
                        const std::string& cpu_code_isotropic,
                        const std::vector<std::string>& compiler_args,
                        Scalar r_cut_isotropic,
                        pybind11::array_t<float> param_array_isotropic,
                        const std::string& cpu_code_constituent,
                        Scalar r_cut_constituent,
                        pybind11::array_t<float> param_array_constituent)
        : PatchEnergyJIT(sysdef,
                         exec_conf,
                         cpu_code_isotropic,
                         compiler_args,
                         r_cut_isotropic,
                         param_array_isotropic,
                         true),
          m_r_cut_constituent(r_cut_constituent),
          m_param_array_constituent(
              param_array_constituent.data(),
              param_array_constituent.data() + param_array_constituent.size(),
              hoomd::detail::managed_allocator<float>(m_exec_conf->isCUDAEnabled()))
        {
        // build the JIT.
        EvalFactory* factory_constituent
            = new EvalFactory(cpu_code_constituent, compiler_args, this->m_is_union);

        // get the evaluator and check for errors
        m_eval_constituent = factory_constituent->getEval();
        if (!m_eval_constituent)
            {
            std::ostringstream s;
            s << "Error compiling JIT code:" << std::endl;
            s << cpu_code_constituent << std::endl;
            s << factory_constituent->getError() << std::endl;
            throw std::runtime_error(s.str());
            }

        factory_constituent->setAlphaUnionArray(&m_param_array_constituent.front());
        m_factory_constituent = std::shared_ptr<EvalFactory>(factory_constituent);

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
    virtual void buildOBBTree(unsigned int type_id);

    //! Set per-type typeid of constituent particles
    virtual void setTypeids(std::string type, pybind11::list typeids)
        {
        unsigned int pid = m_sysdef->getParticleData()->getTypeByName(type);
        auto N = pybind11::len(typeids);
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
        auto N = pybind11::len(position);
        m_position[pid].resize(N);
        for (unsigned int i = 0; i < N; i++)
            {
            pybind11::tuple p_i = position[i];
            vec3<float> pos(p_i[0].cast<float>(), p_i[1].cast<float>(), p_i[2].cast<float>());
            m_position[pid][i] = pos;
            }
        buildOBBTree(pid);
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
        auto N = pybind11::len(orientation);
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
        auto N = pybind11::len(diameter);
        m_diameter[pid].resize(N);
        for (unsigned int i = 0; i < N; i++)
            {
            m_diameter[pid][i] = diameter[i].cast<float>();
            }
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
        auto N = pybind11::len(charge);
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
        // m_build_obb = true;
        // buildOBBTree(); // TODO: loop over all types
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

    //! Override inherited setRCut() to do nothing so that m_r_cut_isotropic doesn't get set
    virtual void setRCut(Scalar r_cut) { }

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
        // buildOBBTree();  // TODO: investigate if this is needed here
        }

    //! Get the cut-off for constituent particles
    virtual Scalar getRCutIsotropic()
        {
        // return cutoff for constituent particle potentials
        return m_r_cut_isotropic;
        }

    //! Set the cut-off for constituent particles
    virtual void setRCutIsotropic(Scalar r_cut)
        {
        // return cutoff for constituent particle potentials
        m_r_cut_isotropic = r_cut;
        // buildOBBTree();  // TODO: investigate if this is needed here
        }

    //! Get the maximum geometric extent, which is added to the cutoff, per type
    virtual inline Scalar getAdditiveCutoff(unsigned int type)
        {
        assert(type <= m_extent_type.size());
        Scalar extent = m_extent_type[type];
        // ensure the minimum cutoff distance is the isotropic r_cut
        if (0.5 * extent + m_r_cut_constituent < m_r_cut_isotropic)
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

    static pybind11::object getParamArrayConstituent(pybind11::object self)
        {
        auto self_cpp = self.cast<PatchEnergyJITUnion*>();
        unsigned int array_size = (unsigned int)self_cpp->m_param_array_constituent.size();
        return pybind11::array(array_size,
                               self_cpp->m_factory_constituent->getAlphaUnionArray(),
                               self);
        }

    protected:
    std::vector<hpmc::detail::GPUTree> m_tree; // The tree acceleration structure per particle type
    std::vector<Scalar> m_extent_type;         // The per-type geometric extent
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
        m_factory_constituent; //!< The factory for the evaluator function, for constituent ptls
    EvalFactory::EvalFnPtr
        m_eval_constituent;     //!< Pointer to evaluator function inside the JIT module
    Scalar m_r_cut_constituent; //!< Cutoff on constituent particles
    std::vector<float, hoomd::detail::managed_allocator<float>>
        m_param_array_constituent; //!< Data array for constituent particles
    };

namespace detail
    {
//! Exports the PatchEnergyJITUnion class to python
void export_PatchEnergyJITUnion(pybind11::module& m);

    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
#endif // _PATCH_ENERGY_JIT_UNION_H_
