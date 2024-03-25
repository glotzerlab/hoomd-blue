// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include <memory>

#include <vector>

/*! \file PotentialSpecialPair.h
    \brief Declares PotentialSpecialPair
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __POTENTIALSPECIAL_PAIR_H__
#define __POTENTIALSPECIAL_PAIR_H__

namespace hoomd
    {
namespace md
    {
/*! SpecialPair potential with evaluator support

    Specific particle pairs can be connected by pair potentials.
    These act much like bonds (see PotentialBond), but they have their own data structure.

    \ingroup computes
*/
template<class evaluator> class PotentialSpecialPair : public ForceCompute
    {
    public:
    //! Param type from evaluator
    typedef typename evaluator::param_type param_type;

    //! Constructs the compute
    PotentialSpecialPair(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~PotentialSpecialPair();

    /// Validate the given type
    virtual void validateType(unsigned int type, std::string action);

    //! Set the parameters
    virtual void setParams(unsigned int type, const param_type& param);

    virtual void setParamsPython(std::string type, pybind11::dict param);

    /// Set the r_cut for a given type
    virtual void setRCut(std::string type, Scalar r_cut);

    /// Get the r_cut for a given type
    virtual Scalar getRCut(std::string type);

    /// Get the parameters for a specific type
    virtual pybind11::dict getParams(std::string type);

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    virtual CommFlags getRequestedCommFlags(uint64_t timestep);
#endif

    protected:
    GPUArray<param_type> m_params;         //!< SpecialPair parameters per type
    std::shared_ptr<PairData> m_pair_data; //!< Data to use in computing particle pairs

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

/*! \param sysdef System to compute forces on
 */
template<class evaluator>
PotentialSpecialPair<evaluator>::PotentialSpecialPair(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing PotentialSpecialPair<" << evaluator::getName()
                                << ">" << std::endl;
    assert(m_pdata);

    // access the pair data for later use
    m_pair_data = m_sysdef->getPairData();

    // allocate the parameters
    GPUArray<param_type> params(m_pair_data->getNTypes(), m_exec_conf);
    m_params.swap(params);
    }

template<class evaluator> PotentialSpecialPair<evaluator>::~PotentialSpecialPair()
    {
    m_exec_conf->msg->notice(5) << "Destroying PotentialSpecialPair<" << evaluator::getName() << ">"
                                << std::endl;
    }

template<class evaluator>
void PotentialSpecialPair<evaluator>::validateType(unsigned int type, std::string action)
    {
    if (type >= m_pair_data->getNTypes())
        {
        std::string err("Invalid pair type specified: ");
        throw std::runtime_error(err + "Error " + action + " in PotentialSpecialPair");
        }
    }

/*! \param type Type of the pair to set parameters for
    \param param Parameter to set

    Sets the parameters for the potential of a particular pair type
*/
template<class evaluator>
void PotentialSpecialPair<evaluator>::setParams(unsigned int type, const param_type& param)
    {
    // make sure the type is valid
    validateType(type, "setting parameters");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[type] = param;
    }

/*! \param type String of the type of the pair to set parameters for
    \param param Parameters to set in a python dictionary

    Sets the parameters for the potential of a particular pair type
*/
template<class evaluator>
void PotentialSpecialPair<evaluator>::setParamsPython(std::string type, pybind11::dict param)
    {
    // TODO getTypeByName validates types already, so this twice validates types
    auto typ = m_pair_data->getTypeByName(type);
    param_type _param(param);
    setParams(typ, _param);
    }

/*! \param type String of the type of the pair to get parameters for

    gets the parameters for the potential of a particular pair type
*/
template<class evaluator>
pybind11::dict PotentialSpecialPair<evaluator>::getParams(std::string type)
    {
    // make sure the type is valid
    auto typ = m_pair_data->getTypeByName(type);
    validateType(typ, "getting parameters");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);
    return h_params.data[typ].asDict();
    }

/*! \param type String of the type of the pair to set r_cut for
    \param r_cut r_cut to set

    Sets the r_cut for the potential of a particular pair type
*/
template<class evaluator>
void PotentialSpecialPair<evaluator>::setRCut(std::string type, Scalar r_cut)
    {
    auto typ = m_pair_data->getTypeByName(type);
    validateType(typ, "setting r_cut");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[typ].r_cutsq = r_cut * r_cut;
    }

/*! \param type String of the type of the pair to get r_cut for

    Gets the r_cut for the potential of a particular pair type
*/
template<class evaluator> Scalar PotentialSpecialPair<evaluator>::getRCut(std::string type)
    {
    auto typ = m_pair_data->getTypeByName(type);
    validateType(typ, "getting r_cut");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);
    return sqrt(h_params.data[typ].r_cutsq);
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
template<class evaluator> void PotentialSpecialPair<evaluator>::computeForces(uint64_t timestep)
    {
    assert(m_pdata);

    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::readwrite);

    // access the parameters
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_charge.data);

    // Zero data for force calculation
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    // we are using the minimum image of the global box here
    // to ensure that ghosts are always correctly wrapped (even if a bond exceeds half the domain
    // length)
    const BoxDim box = m_pdata->getGlobalBox();

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    Scalar bond_virial[6];
    for (unsigned int i = 0; i < 6; i++)
        bond_virial[i] = Scalar(0.0);

    ArrayHandle<typename PairData::members_t> h_bonds(m_pair_data->getMembersArray(),
                                                      access_location::host,
                                                      access_mode::read);
    ArrayHandle<typeval_t> h_typeval(m_pair_data->getTypeValArray(),
                                     access_location::host,
                                     access_mode::read);

    unsigned int max_local = m_pdata->getN() + m_pdata->getNGhosts();

    // for each of the bonds
    const unsigned int size = (unsigned int)m_pair_data->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the bond
        const typename PairData::members_t& bond = h_bonds.data[i];
        assert(bond.tag[0] < m_pdata->getMaximumTag() + 1);
        assert(bond.tag[1] < m_pdata->getMaximumTag() + 1);

        // transform a and b into indices into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[bond.tag[0]];
        unsigned int idx_b = h_rtag.data[bond.tag[1]];

        // throw an error if this bond is incomplete
        if (idx_a >= max_local || idx_b >= max_local)
            {
            this->m_exec_conf->msg->error()
                << "special_pair." << evaluator::getName() << ": bond " << bond.tag[0] << " "
                << bond.tag[1] << " incomplete." << std::endl
                << std::endl;
            throw std::runtime_error("Error in bond calculation");
            }

        // calculate d\vec{r}
        // (MEM TRANSFER: 6 Scalars / FLOPS: 3)
        Scalar3 posa = make_scalar3(h_pos.data[idx_a].x, h_pos.data[idx_a].y, h_pos.data[idx_a].z);
        Scalar3 posb = make_scalar3(h_pos.data[idx_b].x, h_pos.data[idx_b].y, h_pos.data[idx_b].z);

        Scalar3 dx = posb - posa;

        // access charge (if needed)
        Scalar charge_a = Scalar(0.0);
        Scalar charge_b = Scalar(0.0);
        if (evaluator::needsCharge())
            {
            charge_a = h_charge.data[idx_a];
            charge_b = h_charge.data[idx_b];
            }

        // if the vector crosses the box, pull it back
        dx = box.minImage(dx);

        // calculate r_ab squared
        Scalar rsq = dot(dx, dx);

        // get parameters for this bond type
        const param_type& param = h_params.data[h_typeval.data[i].type];

        // compute the force and potential energy
        Scalar force_divr = Scalar(0.0);
        Scalar bond_eng = Scalar(0.0);
        evaluator eval(rsq, param);
        if (evaluator::needsCharge())
            eval.setCharge(charge_a, charge_b);

        bool evaluated = eval.evalForceAndEnergy(force_divr, bond_eng);

        // Bond energy must be halved
        bond_eng *= Scalar(0.5);

        if (evaluated)
            {
            // calculate virial
            if (compute_virial)
                {
                Scalar force_div2r = Scalar(1.0 / 2.0) * force_divr;
                bond_virial[0] = dx.x * dx.x * force_div2r; // xx
                bond_virial[1] = dx.x * dx.y * force_div2r; // xy
                bond_virial[2] = dx.x * dx.z * force_div2r; // xz
                bond_virial[3] = dx.y * dx.y * force_div2r; // yy
                bond_virial[4] = dx.y * dx.z * force_div2r; // yz
                bond_virial[5] = dx.z * dx.z * force_div2r; // zz
                }

            // add the force to the particles (only for non-ghost particles)
            if (idx_b < m_pdata->getN())
                {
                h_force.data[idx_b].x += force_divr * dx.x;
                h_force.data[idx_b].y += force_divr * dx.y;
                h_force.data[idx_b].z += force_divr * dx.z;
                h_force.data[idx_b].w += bond_eng;
                if (compute_virial)
                    for (unsigned int i = 0; i < 6; i++)
                        h_virial.data[i * m_virial_pitch + idx_b] += bond_virial[i];
                }

            if (idx_a < m_pdata->getN())
                {
                h_force.data[idx_a].x -= force_divr * dx.x;
                h_force.data[idx_a].y -= force_divr * dx.y;
                h_force.data[idx_a].z -= force_divr * dx.z;
                h_force.data[idx_a].w += bond_eng;
                if (compute_virial)
                    for (unsigned int i = 0; i < 6; i++)
                        h_virial.data[i * m_virial_pitch + idx_a] += bond_virial[i];
                }
            }
        else
            {
            this->m_exec_conf->msg->error()
                << "special_pair." << evaluator::getName() << ": bond out of bounds" << std::endl
                << std::endl;
            throw std::runtime_error("Error in special pair calculation");
            }
        }
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template<class evaluator>
CommFlags PotentialSpecialPair<evaluator>::getRequestedCommFlags(uint64_t timestep)
    {
    CommFlags flags = CommFlags(0);

    flags[comm_flag::tag] = 1;

    if (evaluator::needsCharge())
        flags[comm_flag::charge] = 1;

    flags |= ForceCompute::getRequestedCommFlags(timestep);

    return flags;
    }
#endif

namespace detail
    {
//! Exports the PotentialSpecialPair class to python
/*! \param name Name of the class in the exported python module
    \tparam T evaluator type to export.
*/
template<class T> void export_PotentialSpecialPair(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialSpecialPair<T>,
                     ForceCompute,
                     std::shared_ptr<PotentialSpecialPair<T>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &PotentialSpecialPair<T>::setParamsPython)
        .def("getParams", &PotentialSpecialPair<T>::getParams)
        .def("setRCut", &PotentialSpecialPair<T>::setRCut)
        .def("getRCut", &PotentialSpecialPair<T>::getRCut);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
