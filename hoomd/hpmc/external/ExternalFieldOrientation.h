// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __EXTERNAL_FIELD_ORIENTATION_H__
#define __EXTERNAL_FIELD_ORIENTATION_H__

#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"
#include "hoomd/VectorMath.h"
#include "hoomd/hpmc/ExternalPotential.h"
#include <pybind11/pybind11.h>

#ifndef __HIPCC__
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#endif

/*! \file ExternalFieldOrientation.h
    \brief Declares an external field that penalizes particle orientations in boundary regions
    
    This field is designed for liquid crystal bend deformation simulations following
    de Pablo group methodology (Soft Matter, 2014, DOI: 10.1039/C3SM51919H).
    
    Particles in the left and right boundary regions (x < -Lx/3 or x > +Lx/3) are
    energetically penalized for misalignment with the z-axis.
*/

namespace hoomd
    {
namespace hpmc
    {

//! Orientation-dependent external field for boundary constraint
/*! This external field applies an energy penalty to particles that are misaligned
    with a target orientation (typically z-axis) when located in boundary regions.
    
    Energy function:
    E_i = kappa * (1 - |q_i · q_target|)²   if x_i in boundary region
    E_i = 0                                  otherwise
    
    where:
    - q_i is the particle's quaternion orientation
    - q_target is the target quaternion ([1,0,0,0] for z-alignment)
    - kappa is the alignment strength in units of kT
    - boundary regions are defined as x < -Lx/3 or x > +Lx/3
    
    This creates a "soft wall" for orientations while still allowing hard sphere
    overlap checking to reject invalid moves.
*/
template<class Shape> class ExternalFieldOrientation : public ExternalPotential
    {
    public:
    //! Constructor
    /*! \param sysdef System definition
        \param kappa Alignment strength in units of kT
        \param target_quat_array Target orientation quaternion as [w,x,y,z] (default: [1,0,0,0] for z-axis)
        \param symmetries_array Symmetrically equivalent orientations (N_sym x 4 array of quaternions)
        \param field_axis Axis along which to apply field (0=x, 1=y, 2=z)
        \param field_extent Fraction of box that constitutes boundary regions (0.0 to 0.5)
    */
    ExternalFieldOrientation(std::shared_ptr<SystemDefinition> sysdef,
                             Scalar kappa,
                             const pybind11::array_t<double>& target_quat_array,
                             const pybind11::array_t<double>& symmetries_array,
                             unsigned int field_axis = 0,
                             Scalar field_extent = Scalar(1.0/6.0))
        : ExternalPotential(sysdef), m_kappa(kappa), m_field_axis(field_axis), m_field_extent(field_extent)
        {
        // Validate field_axis
        if (field_axis > 2)
            {
            throw std::runtime_error("field_axis must be 0 (x), 1 (y), or 2 (z)");
            }
        
        // Validate field_extent
        if (field_extent < 0.0 || field_extent > 0.5)
            {
            throw std::runtime_error("field_extent must be between 0.0 and 0.5");
            }
        
        // Convert target quaternion numpy array to quat
        if (target_quat_array.size() != 4)
            {
            throw std::runtime_error("Target quaternion must have exactly 4 elements [w, x, y, z]");
            }
        const double* data = static_cast<const double*>(target_quat_array.data());
        m_target_quat = quat<Scalar>(data[0], vec3<Scalar>(data[1], data[2], data[3]));
        
        // Convert symmetries array
        setSymmetricallyEquivalentOrientations(symmetries_array);
        }

    protected:
    //! Compute the energy for one particle
    /*! \param timestep Current simulation timestep
        \param tag_i Particle tag
        \param type_i Particle type index
        \param r_i Position of particle i
        \param q_i Orientation quaternion of particle i
        \param charge_i Particle charge (unused for this field)
        \param trial Trial move indicator
        \returns Energy in units of kT
    */
    virtual LongReal particleEnergyImplementation(uint64_t timestep,
                                                  unsigned int tag_i,
                                                  unsigned int type_i,
                                                  const vec3<LongReal>& r_i,
                                                  const quat<LongReal>& q_i,
                                                  LongReal charge_i,
                                                  Trial trial = Trial::None) override
        {
        // Get box dimensions
        const auto& particle_data = m_sysdef->getParticleData();
        auto box = particle_data->getGlobalBox();
        
        // Get box length along the field axis
        LongReal L_axis;
        LongReal pos_along_axis;
        
        if (m_field_axis == 0)  // x-axis
            {
            L_axis = box.getL().x;
            pos_along_axis = r_i.x;
            }
        else if (m_field_axis == 1)  // y-axis
            {
            L_axis = box.getL().y;
            pos_along_axis = r_i.y;
            }
        else  // z-axis (m_field_axis == 2)
            {
            L_axis = box.getL().z;
            pos_along_axis = r_i.z;
            }
        
        // Define boundary regions based on field_extent
        // field_extent = 1/6 means: left 1/6 and right 1/6 are boundaries
        // Left boundary: pos < -L/2 + L*field_extent
        // Right boundary: pos > L/2 - L*field_extent
        LongReal boundary_width = L_axis * LongReal(m_field_extent);
        LongReal left_boundary = -L_axis / LongReal(2.0) + boundary_width;
        LongReal right_boundary = L_axis / LongReal(2.0) - boundary_width;
        
        // Check if particle is in boundary region and calculate distance-based attenuation
        LongReal attenuation_factor = LongReal(0.0);
        
        if (pos_along_axis < left_boundary)
            {
            // In left boundary region - calculate distance from left wall
            LongReal distance_from_wall = pos_along_axis - (-L_axis / LongReal(2.0));
            // Add small offset to avoid division by zero at the wall
            LongReal r = distance_from_wall + LongReal(1e-6);
            // 1/r² attenuation, normalized so that attenuation = 1 at distance = 1e-6
            // and attenuation → 0 as distance → boundary_width
            LongReal r_ref = LongReal(1e-6);  // Reference distance where attenuation = 1
            attenuation_factor = (r_ref * r_ref) / (r * r);
            // Smoothly cut off at boundary edge
            if (distance_from_wall >= boundary_width) attenuation_factor = LongReal(0.0);
            }
        else if (pos_along_axis > right_boundary)
            {
            // In right boundary region - calculate distance from right wall
            LongReal distance_from_wall = (L_axis / LongReal(2.0)) - pos_along_axis;
            // Add small offset to avoid division by zero at the wall
            LongReal r = distance_from_wall + LongReal(1e-6);
            // 1/r² attenuation
            LongReal r_ref = LongReal(1e-6);  // Reference distance where attenuation = 1
            attenuation_factor = (r_ref * r_ref) / (r * r);
            // Smoothly cut off at boundary edge
            if (distance_from_wall >= boundary_width) attenuation_factor = LongReal(0.0);
            }
        else
            {
            return LongReal(0.0);  // No energy in central region
            }
        
        // Clamp attenuation factor for numerical safety (should already be > 0 due to offset)
        if (attenuation_factor < LongReal(0.0)) attenuation_factor = LongReal(0.0);
        
        // Find maximum alignment with target orientation considering all symmetries
        // For each symmetry, compute the effective target as target * symmetry
        // and find the best alignment
        LongReal max_alignment = LongReal(0.0);
        
        for (size_t i_sym = 0; i_sym < m_symmetry.size(); ++i_sym)
            {
            // Compute rotated target: q_target_effective = q_target * q_symmetry
            quat<LongReal> target_long(m_target_quat.s, vec3<LongReal>(m_target_quat.v));
            quat<LongReal> sym_long(m_symmetry[i_sym].s, vec3<LongReal>(m_symmetry[i_sym].v));
            quat<LongReal> q_target_effective = target_long * sym_long;
            
            // Calculate quaternion dot product with this symmetry-rotated target
            LongReal dot = q_i.s * q_target_effective.s
                         + q_i.v.x * q_target_effective.v.x
                         + q_i.v.y * q_target_effective.v.y
                         + q_i.v.z * q_target_effective.v.z;
            
            // Use absolute value (both aligned and anti-aligned are equivalent)
            LongReal alignment = (dot < LongReal(0.0)) ? -dot : dot;
            
            // Keep track of best alignment
            if (alignment > max_alignment)
                {
                max_alignment = alignment;
                }
            }
        
        // Energy penalty for misalignment with 1/r² distance-based attenuation
        // E = kappa * attenuation_factor * (1 - max_alignment)²
        // attenuation_factor: decays as 1/r² from wall, with attenuation ≈ 1 near wall
        LongReal misalignment = LongReal(1.0) - max_alignment;
        LongReal energy = LongReal(m_kappa) * attenuation_factor * misalignment * misalignment;
        
        return energy;
        }

    public:
    //! Set the alignment strength
    void setKappa(Scalar kappa)
        {
        m_kappa = kappa;
        }

    //! Get the alignment strength
    Scalar getKappa() const
        {
        return m_kappa;
        }

    //! Set the target orientation from numpy array
    void setTargetOrientation(const pybind11::array_t<double>& target_quat_array)
        {
        if (target_quat_array.size() != 4)
            {
            throw std::runtime_error("Target quaternion must have exactly 4 elements [w, x, y, z]");
            }
        const double* data = static_cast<const double*>(target_quat_array.data());
        m_target_quat = quat<Scalar>(data[0], vec3<Scalar>(data[1], data[2], data[3]));
        }

    //! Get the target orientation as array
    pybind11::array_t<double> getTargetOrientation() const
        {
        pybind11::array_t<double> result(4);
        double* data = static_cast<double*>(result.mutable_data());
        data[0] = m_target_quat.s;
        data[1] = m_target_quat.v.x;
        data[2] = m_target_quat.v.y;
        data[3] = m_target_quat.v.z;
        return result;
        }

    //! Set symmetrically equivalent orientations from a (N_symmetry, 4) numpy array
    void setSymmetricallyEquivalentOrientations(const pybind11::array_t<double>& equivalent_quaternions)
        {
        if (equivalent_quaternions.ndim() != 2)
            {
            throw std::runtime_error("Symmetries array must be 2D with shape (N_sym, 4).");
            }

        const size_t N_sym = equivalent_quaternions.shape(0);
        const size_t dim = equivalent_quaternions.shape(1);
        if (dim != 4)
            {
            throw std::runtime_error("Symmetries array must have 4 columns [w, x, y, z].");
            }
        
        const double* rawdata = static_cast<const double*>(equivalent_quaternions.data());
        m_symmetry.resize(N_sym);
        for (size_t i = 0; i < N_sym * 4; i += 4)
            {
            m_symmetry[i / 4] = quat<Scalar>(rawdata[i],
                                             vec3<Scalar>(rawdata[i + 1], rawdata[i + 2], rawdata[i + 3]));
            }
        }

    //! Get symmetrically equivalent orientations as numpy array
    pybind11::array_t<double> getSymmetricallyEquivalentOrientations() const
        {
        const size_t N_sym = m_symmetry.size();
        pybind11::array_t<double> result({N_sym, size_t(4)});
        double* data = static_cast<double*>(result.mutable_data());
        
        for (size_t i = 0; i < N_sym; ++i)
            {
            data[i * 4 + 0] = m_symmetry[i].s;
            data[i * 4 + 1] = m_symmetry[i].v.x;
            data[i * 4 + 2] = m_symmetry[i].v.y;
            data[i * 4 + 3] = m_symmetry[i].v.z;
            }
        
        return result;
        }

    //! Set the field axis (0=x, 1=y, 2=z)
    void setFieldAxis(unsigned int axis)
        {
        if (axis > 2)
            {
            throw std::runtime_error("field_axis must be 0 (x), 1 (y), or 2 (z)");
            }
        m_field_axis = axis;
        }

    //! Get the field axis
    unsigned int getFieldAxis() const
        {
        return m_field_axis;
        }

    //! Set the field extent (fraction of box that is boundary region)
    void setFieldExtent(Scalar extent)
        {
        if (extent < 0.0 || extent > 0.5)
            {
            throw std::runtime_error("field_extent must be between 0.0 and 0.5");
            }
        m_field_extent = extent;
        }

    //! Get the field extent
    Scalar getFieldExtent() const
        {
        return m_field_extent;
        }

#ifndef __HIPCC__
    //! Get the python class name
    static std::string getName()
        {
        return std::string("OrientationField");
        }
#endif

    protected:
    Scalar m_kappa;                       //!< Alignment strength in kT
    quat<Scalar> m_target_quat;           //!< Target orientation quaternion
    std::vector<quat<Scalar>> m_symmetry; //!< Symmetrically equivalent orientations
    unsigned int m_field_axis;            //!< Axis along which to apply field (0=x, 1=y, 2=z)
    Scalar m_field_extent;                //!< Fraction of box that constitutes boundary regions
    };

namespace detail
    {
//! Export the ExternalFieldOrientation class to python
template<class Shape>
void export_ExternalFieldOrientation(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ExternalFieldOrientation<Shape>,
                     ExternalPotential,
                     std::shared_ptr<ExternalFieldOrientation<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            Scalar,
                            const pybind11::array_t<double>&,
                            const pybind11::array_t<double>&,
                            unsigned int,
                            Scalar>(),
             pybind11::arg("sysdef"),
             pybind11::arg("kappa"),
             pybind11::arg("target_quat"),
             pybind11::arg("symmetries"),
             pybind11::arg("field_axis") = 0,
             pybind11::arg("field_extent") = Scalar(1.0/6.0))
        .def_property("kappa",
                      &ExternalFieldOrientation<Shape>::getKappa,
                      &ExternalFieldOrientation<Shape>::setKappa)
        .def_property("target_orientation",
                      &ExternalFieldOrientation<Shape>::getTargetOrientation,
                      &ExternalFieldOrientation<Shape>::setTargetOrientation)
        .def_property("symmetries",
                      &ExternalFieldOrientation<Shape>::getSymmetricallyEquivalentOrientations,
                      &ExternalFieldOrientation<Shape>::setSymmetricallyEquivalentOrientations)
        .def_property("field_axis",
                      &ExternalFieldOrientation<Shape>::getFieldAxis,
                      &ExternalFieldOrientation<Shape>::setFieldAxis)
        .def_property("field_extent",
                      &ExternalFieldOrientation<Shape>::getFieldExtent,
                      &ExternalFieldOrientation<Shape>::setFieldExtent);
    }
    } // end namespace detail

    } // end namespace hpmc
    } // end namespace hoomd

#endif // __EXTERNAL_FIELD_ORIENTATION_H__
