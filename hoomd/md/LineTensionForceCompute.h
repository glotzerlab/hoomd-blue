// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "MeshForceCompute.h"
#include "HelfrichMeshParameters.h"

#include <memory>

/*! \file LineTensionForceCompute.h
    \brief Declares a class for computing line tension forces
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __LINETENSIONFORCECOMPUTE_H__
#define __LINETENSIONFORCECOMPUTE_H__

namespace hoomd
{
namespace md
{

//! Parameters for line tension potential
struct line_tension_param_t
{
    Scalar l;

    line_tension_param_t()
        : l(0) {}

    line_tension_param_t(pybind11::dict params)
    {
		l = pybind11::cast<Scalar>(params["l"]);
	}

    pybind11::dict asDict() const
    {
        pybind11::dict d;
        d["l"] = l;
        return d;
    }
};

//! Computes line tension forces on mesh interfaces
class PYBIND11_EXPORT LineTensionForceCompute
    : public MeshForceCompute
{
	public:
    //! Constructor
	LineTensionForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                             	std::shared_ptr<MeshDefinition> meshdef);
    
	//! Destructor
    virtual ~LineTensionForceCompute();

    /*! \param typ1 First type index in the pair
    	\param typ2 Second type index in the pair
    	\param param Parameter to set
    	\note When setting the value for (\a typ1, \a typ2), the parameter for (\a typ2, \a typ1) is
   		automatically set.
	*/
	//! Set parameters
    virtual void setParams(
        unsigned int typ1,
		unsigned int typ2,
        const line_tension_param_t& params
		);


    //! Set parameters from Python
	virtual void setParamsPython(pybind11::tuple typ, pybind11::dict params);
	
    //! Get parameters for single type pair using tuple of strings
	pybind11::dict getParams(pybind11::tuple typ);
	

#ifdef ENABLE_MPI
    CommFlags getRequestedCommFlags(
        uint64_t timestep) override
    {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |=
            ForceCompute::getRequestedCommFlags(
                timestep);
        return flags;
    }
#endif


protected:
	// Per type pair potential parameters
	//vector<param_type, hoomd::detail::managed_allocator<param_type>> m_params;
	
	Index2D m_typpair_idx;        //!< Helper class for indexing per type pair arrays
    GPUArray<line_tension_param_t> m_params;
	GPUArray<Scalar> m_particle_energy;

    //! Compute forces
    void computeForces(
        uint64_t timestep) override;

//Included later in PairPotential.h
/*
 m_params = std::vector<param_type, hoomd::detail::managed_allocator<param_type>>(          
        m_typpair_idx.getNumElements(),
        param_type(),
        hoomd::detail::managed_allocator<param_type>(m_exec_conf->isCUDAEnabled()));
*/
	};


namespace detail
{
void export_LineTensionForceCompute(
    pybind11::module& m);
}

} // namespace md
} // namespace hoomd

#endif
