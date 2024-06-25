// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/VirtualParticleFiller.h
 * \brief Definition of abstract base class for backfilling solid boundaries with virtual particles.
 */

#ifndef MPCD_VIRTUAL_PARTICLE_FILLER_H_
#define MPCD_VIRTUAL_PARTICLE_FILLER_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "CellList.h"

#include "hoomd/Autotuned.h"
#include "hoomd/SystemDefinition.h"
#include "hoomd/Variant.h"
#include <pybind11/pybind11.h>

#include <string>

namespace hoomd
    {
namespace mpcd
    {
//! Adds virtual particles to the MPCD particle data
/*!
 * Virtual particles are used to pad cells sliced by solid boundaries so that their viscosity does
 * not get too low. The VirtualParticleFiller base class defines an interface for adding these
 * particles. Deriving classes must implement a fill() method that adds the particles.
 */
class PYBIND11_EXPORT VirtualParticleFiller : public Autotuned
    {
    public:
    VirtualParticleFiller(std::shared_ptr<SystemDefinition> sysdef,
                          const std::string& type,
                          Scalar density,
                          std::shared_ptr<Variant> T);

    virtual ~VirtualParticleFiller() { }

    //! Fill up virtual particles
    virtual void fill(uint64_t timestep) { }

    //! Get the fill particle density
    Scalar getDensity() const
        {
        return m_density;
        }

    //! Set the fill particle density
    void setDensity(Scalar density);

    //! Get the filler particle type
    std::string getType() const;

    //! Set the fill particle type
    void setType(const std::string& type);

    //! Get the fill particle temperature
    std::shared_ptr<Variant> getTemperature() const
        {
        return m_T;
        }

    //! Set the fill particle temperature
    void setTemperature(std::shared_ptr<Variant> T)
        {
        m_T = T;
        }

    //! Set the cell list used for filling
    virtual void setCellList(std::shared_ptr<mpcd::CellList> cl)
        {
        m_cl = cl;
        }

    protected:
    std::shared_ptr<SystemDefinition> m_sysdef;                //!< HOOMD system definition
    std::shared_ptr<hoomd::ParticleData> m_pdata;              //!< HOOMD particle data
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Execution configuration
    std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata;          //!< MPCD particle data
    std::shared_ptr<mpcd::CellList> m_cl;                      //!< MPCD cell list

    Scalar m_density;             //!< Fill density
    unsigned int m_type;          //!< Fill type
    std::shared_ptr<Variant> m_T; //!< Temperature for filled particles
    unsigned int m_filler_id;     //!< Unique ID of this filler

    unsigned int computeFirstTag(unsigned int N_fill) const;

    private:
    static unsigned int s_filler_count; //!< Total count of fillers, used to assign unique ID
    };

namespace detail
    {
//! Export the VirtualParticleFiller to python
void export_VirtualParticleFiller(pybind11::module& m);
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_VIRTUAL_PARTICLE_FILLER_H_
