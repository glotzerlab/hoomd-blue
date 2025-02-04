// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PPPM_FORCE_COMPUTE_H__
#define __PPPM_FORCE_COMPUTE_H__

#include "NeighborList.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/ParticleGroup.h"

#ifdef ENABLE_MPI
#include "CommunicatorGrid.h"
#include "hoomd/extern/dfftlib/src/dfft_host.h"
#endif

#include "hoomd/extern/kiss_fftnd.h"

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <memory>

namespace hoomd
    {
namespace md
    {
const Scalar EPS_HOC(1.0e-7);
const unsigned int PPPM_MAX_ORDER = 7;

/*! Compute the long-ranged part of the particle-particle particle-mesh Ewald sum (PPPM)
 */
class PYBIND11_EXPORT PPPMForceCompute : public ForceCompute
    {
    public:
    //! Constructor
    PPPMForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<NeighborList> nlist,
                     std::shared_ptr<ParticleGroup> group);
    virtual ~PPPMForceCompute();

    //! Set the parameters
    virtual void setParams(unsigned int nx,
                           unsigned int ny,
                           unsigned int nz,
                           unsigned int order,
                           Scalar kappa,
                           Scalar rcut,
                           Scalar alpha = 0);

    void computeForces(uint64_t timestep);

    //! Get sum of charges
    Scalar getQSum();

    //! Get sum of squares of charges
    Scalar getQ2Sum();

    /// Get the grid resolution
    pybind11::tuple getResolution()
        {
        pybind11::list val;
        val.append(m_global_dim.x);
        val.append(m_global_dim.y);
        val.append(m_global_dim.z);

        return pybind11::tuple(val);
        }

    unsigned int getOrder()
        {
        return m_order;
        }

    Scalar getKappa()
        {
        return m_kappa;
        }

    Scalar getRCut()
        {
        return m_rcut;
        }

    Scalar getAlpha()
        {
        return m_alpha;
        }

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    /*! \param timestep Current time step
     */
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = ForceCompute::getRequestedCommFlags(timestep);
        bool correct_body = m_nlist->getFilterBody();

        if (m_nlist->getExclusionsSet() || correct_body)
            {
            // need ghost particle charge
            flags[comm_flag::charge] = 1;

            if (correct_body)
                {
                flags[comm_flag::body] = 1;
                }
            }

        return flags;
        }
#endif

    protected:
    /*! Compute the biased forces for this collective variable.
        The force that is written to the force arrays must be
        multiplied by the bias factor.

        \param timestep The current value of the time step
     */
    void computeBiasForces(uint64_t timestep);

    std::shared_ptr<NeighborList> m_nlist;  //!< The neighborlist to use for the computation
    std::shared_ptr<ParticleGroup> m_group; //!< Group to compute properties for

    uint3 m_mesh_points;          //!< Number of sub-divisions along one coordinate
    uint3 m_global_dim;           //!< Global grid dimensions
    uint3 m_n_ghost_cells;        //!< Number of ghost cells along every axis
    uint3 m_grid_dim;             //!< Grid dimensions (including ghost cells)
    Scalar3 m_ghost_width;        //!< Dimensions of the ghost layer
    unsigned int m_ghost_offset;  //!< Offset in mesh due to ghost cells
    unsigned int m_n_cells;       //!< Total number of inner cells
    unsigned int m_radius;        //!< Stencil radius (in units of mesh size)
    unsigned int m_n_inner_cells; //!< Number of inner mesh points (without ghost cells)
    GlobalArray<Scalar> m_inf_f;  //!< Fourier representation of the influence function (real part)
    GlobalArray<Scalar3> m_k;     //!< Mesh of k values
    Scalar m_qstarsq;             //!< Short wave length cut-off squared for density harmonics
    bool m_need_initialize;       //!< True if we have not yet computed the influence function
    bool m_params_set;            //!< True if parameters are set
    bool m_box_changed;           //!< True if box has changed since last compute

    GlobalArray<Scalar> m_virial_mesh; //!< k-space mesh of virial tensor values

    Scalar m_kappa; //!< Splitting parameter
    Scalar m_rcut;  //!< Cutoff for short-ranged interaction
    int m_order;    //!< Order of interpolation scheme
    Scalar m_alpha; //!< Debye screening parameter

    Scalar m_q;  //!< Total system charge
    Scalar m_q2; //!< Sum of charge squared

    GlobalArray<Scalar> m_rho_coeff; //!< Coefficients for computing the grid based charge density
    GlobalArray<Scalar> m_gf_b;      //!< Green function coefficients

    Scalar m_body_energy;      //!< Energy correction due to rigid body exclusions
    bool m_ptls_added_removed; //!< True if global particle number changed

    //! Helper function to be called when particle number changes
    void slotGlobalParticleNumberChange()
        {
        m_ptls_added_removed = true;
        }

    //! Helper function to be called when box changes
    void setBoxChange()
        {
        m_box_changed = true;
        }

    //! Helper function to setup the mesh indices
    void setupMesh();

    //! Helper function to setup FFT and allocate the mesh arrays
    virtual void initializeFFT();

    //! Compute the optimal influence function
    virtual void computeInfluenceFunction();

    //! Helper function to assign particle coordinates to mesh
    virtual void assignParticles();

    //! Helper function to update the mesh arrays
    virtual void updateMeshes();

    //! Helper function to interpolate the forces
    virtual void interpolateForces();

    //! Helper function to calculate value of potential energy
    virtual Scalar computePE();

    //! Helper function to compute the virial
    virtual void computeVirial();

    //! Helper function to correct forces on excluded particles
    virtual void fixExclusions();

    //! Setup coefficients
    virtual void setupCoeffs();

    //! Compute rigid body correction
    virtual void computeBodyCorrection();

    private:
    kiss_fftnd_cfg m_kiss_fft = NULL;  //!< The FFT configuration
    kiss_fftnd_cfg m_kiss_ifft = NULL; //!< Inverse FFT configuration

#ifdef ENABLE_MPI
    dfft_plan m_dfft_plan_forward; //!< Distributed FFT for forward transform
    dfft_plan m_dfft_plan_inverse; //!< Distributed FFT for inverse transform
    std::unique_ptr<CommunicatorGrid<kiss_fft_cpx>>
        m_grid_comm_forward; //!< Communicator for charge mesh
    std::unique_ptr<CommunicatorGrid<kiss_fft_cpx>>
        m_grid_comm_reverse; //!< Communicator for inv fourier mesh
#endif

    bool m_kiss_fft_initialized; //!< True if a local KISS FFT has been set up

    GlobalArray<kiss_fft_cpx> m_mesh;         //!< The particle density mesh
    GlobalArray<kiss_fft_cpx> m_fourier_mesh; //!< The fourier transformed mesh
    GlobalArray<kiss_fft_cpx>
        m_fourier_mesh_G_x; //!< Fourier transformed mesh times the influence function, x-component
    GlobalArray<kiss_fft_cpx>
        m_fourier_mesh_G_y; //!< Fourier transformed mesh times the influence function, y-component
    GlobalArray<kiss_fft_cpx>
        m_fourier_mesh_G_z; //!< Fourier transformed mesh times the influence function, z-component
    GlobalArray<kiss_fft_cpx> m_inv_fourier_mesh_x; //!< Fourier transformed mesh times the
                                                    //!< influence function, x-component
    GlobalArray<kiss_fft_cpx> m_inv_fourier_mesh_y; //!< Fourier transformed mesh times the
                                                    //!< influence function, y-component
    GlobalArray<kiss_fft_cpx> m_inv_fourier_mesh_z; //!< Fourier transformed mesh times the
                                                    //!< influence function, z-component

    bool m_dfft_initialized; //! True if host dfft has been initialized

    //! Compute virial on mesh
    void computeVirialMesh();

    //! Compute number of ghost cellso
    uint3 computeGhostCellNum();

    //! root mean square error in force calculation
    Scalar rms(Scalar h, Scalar prd, Scalar natoms);

    //! computes coefficients for assigning charges to grid points
    void compute_rho_coeff();

    //! computes auxiliary table for optimized influence function
    void compute_gf_denom();

    //! computes coefficients for the Green's function
    Scalar gf_denom(Scalar x, Scalar y, Scalar z);
    };

    } // end namespace md
    } // end namespace hoomd

#endif
