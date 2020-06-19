// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// inclusion guard
#ifndef _EXTERNAL_FIELD_COMPOSITE_H_
#define _EXTERNAL_FIELD_COMPOSITE_H_

/*! \file ExternalField.h
    \brief Declaration of ExternalField base class
*/

#include "hoomd/Compute.h"
#include "hoomd/VectorMath.h"

#include "ExternalField.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

namespace hpmc
{

template< class Shape >
class ExternalFieldMonoComposite : public ExternalFieldMono<Shape>
    {
    public:
        ExternalFieldMonoComposite(std::shared_ptr<SystemDefinition> sysdef) : ExternalFieldMono<Shape>(sysdef) {}

        ~ExternalFieldMonoComposite() {}

        Scalar calculateBoltzmannWeight(unsigned int timestep) { return 0.0; }

        double calculateDeltaE(const Scalar4 * const  position_old,
                                        const Scalar4 * const  orientation_old,
                                        const BoxDim * const  box_old
                                        )
            {
                throw(std::runtime_error("ExternalFieldMonoComposite::calculateDeltaE is not implemented"));
                return double(0.0);
            }

        double energydiff(const unsigned int& index, const vec3<Scalar>& position_old, const Shape& shape_old, const vec3<Scalar>& position_new, const Shape& shape_new)
            {
            double Energy = 0.0;
            for(size_t i = 0; i < m_externals.size(); i++)
                {
                Energy += m_externals[i]->energydiff(index, position_old, shape_old, position_new, shape_new);
                }
            return Energy;
            }

        void addExternal(std::shared_ptr< ExternalFieldMono<Shape> > ext) { m_externals.push_back(ext); }

        void reset(unsigned int timestep)
        {
            for(size_t i = 0; i < m_externals.size(); i++)
            {
                m_externals[i]->reset(timestep);
            }
        }

    private:
        std::vector< std::shared_ptr< ExternalFieldMono<Shape> > > m_externals;
    };



template<class Shape>
void export_ExternalFieldComposite(pybind11::module& m, std::string name)
{
   pybind11::class_<ExternalFieldMonoComposite<Shape>, ExternalFieldMono<Shape>, std::shared_ptr< ExternalFieldMonoComposite<Shape> > >(m, name.c_str())
    .def(pybind11::init< std::shared_ptr<SystemDefinition> >())
    .def("addExternal", &ExternalFieldMonoComposite<Shape>::addExternal)
    ;

}


} // namespace
#endif // inclusion guard
