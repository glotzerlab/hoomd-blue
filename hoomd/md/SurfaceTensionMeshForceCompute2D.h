#include "hoomd/ForceCompute.h"
#include "hoomd/MeshDefinition.h"
#include "SurfaceTensionMeshForceCompute.h"
#include <memory>

#include <vector>

/*! \file SurfaceTensionForceCompute2D.h
    \brief Declares a class for computing surface tension forces in two dimensions
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __SURFACETENSIONMESHFORCECOMPUTE2D_H__
#define __SURFACETENSIONMESHFORCECOMPUTE2D_H__

namespace hoomd
    {
namespace md
    {
//! Computes surface tension forces on the mesh
/*! Surface tension forces are computed on every interface bond in a mesh.
    \ingroup computes
*/
class PYBIND11_EXPORT SurfaceTensionMeshForceCompute2D : public ForceCompute
    {
    public:
    //! Constructs the compute
    SurfaceTensionMeshForceCompute2D(std::shared_ptr<SystemDefinition> sysdef,
                                             std::shared_ptr<MeshDefinition> meshdef);

    //! Destructor
    virtual ~SurfaceTensionMeshForceCompute2D();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar sigma);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a type
    pybind11::dict getParams(std::string type);

    virtual Scalar getCircumference()
        {
        return m_circumference;
        };

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    /*! \param timestep Current time step
     */
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:
    Scalar* m_sigma; //!< sigma surface tension term
    Scalar m_circumference;

    std::shared_ptr<MeshDefinition>
        m_mesh_data; //!< Mesh data to use in computing area conservation energy

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    virtual Scalar energyDiff(unsigned int idx_a,
                              unsigned int idx_b,
                              unsigned int idx_c,
                              unsigned int idx_d,
                              unsigned int type_id);
    };

namespace detail
    {
//! Exports the SurfaceTensionMeshForceCompute2D class to python
void export_SurfaceTensionMeshForceCompute2D(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
