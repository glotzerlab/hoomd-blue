// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "ExternalPotential.h"

namespace hoomd
    {
namespace hpmc
    {
/** Linear potential based on the distance to a plane.

    Use for e.g. gravity.
*/
class ExternalPotentialLinear : public ExternalPotential
    {
    public:
    ExternalPotentialLinear(std::shared_ptr<SystemDefinition> sysdef)
        : ExternalPotential(sysdef), m_alpha(sysdef->getParticleData()->getNTypes())
        {
        m_plane_origin = vec3<LongReal>(0, 0, 0);
        m_plane_normal = vec3<LongReal>(0, 1, 0);
        }
    virtual ~ExternalPotentialLinear() { }

    /// Set the origin of the potential.
    void setPlaneOrigin(const vec3<LongReal>& plane_origin)
        {
        m_plane_origin = plane_origin;
        }

    /// Get the origin.
    vec3<LongReal> getPlaneOrigin() const
        {
        return m_plane_origin;
        }
    /// Set the normal of the potential.
    void setPlaneNormal(const vec3<LongReal>& plane_normal)
        {
        m_plane_normal = normalize(plane_normal);
        }

    /// Get the normal.
    vec3<LongReal> getPlaneNormal() const
        {
        return m_plane_normal;
        }

    /// Set the linear coefficient
    void setAlpha(const std::string& particle_type, LongReal alpha);

    /// Get the linear coefficient.
    LongReal getAlpha(const std::string& particle_type);

    protected:
    /// A point on the plane.
    vec3<LongReal> m_plane_origin;

    /// The normal vector (unit length) perpendicular to the plane.
    vec3<LongReal> m_plane_normal;

    /// The linear coefficient per particle type.
    std::vector<LongReal> m_alpha;

    /** Implement the evaluation the energy of the external field interacting with one particle.

        @param type_i Type index of the particle.
        @param r_i Posiion of the particle in the box.
        @param q_i Orientation of the particle
        @param charge Charge of the particle.
        @param trial Set to false when evaluating the energy of a current configuration. Set to
               true when evaluating a trial move.
        @returns Energy of the external interaction (possibly INFINITY).

        Evaluate the linear potential energy.
    */
    virtual LongReal particleEnergyImplementation(unsigned int type_i,
                                                  const vec3<LongReal>& r_i,
                                                  const quat<LongReal>& q_i,
                                                  LongReal charge_i,
                                                  bool trial);
    };

    } // end namespace hpmc

    } // end namespace hoomd
