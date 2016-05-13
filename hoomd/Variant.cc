// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

/*! \file Variant.cc
    \brief Defines Variant and related classes
*/


#include "Variant.h"

#include <iostream>
#include <stdexcept>
#include <boost/python.hpp>
using namespace boost::python;
using namespace std;

VariantLinear::VariantLinear()
    {
    // initialize m_a and m_b to the end
    m_a = m_values.end();
    m_b = m_values.end();
    }

/*! \param timestep Time step to set the value at
    \param val Value to set at \a timestep

    If a point at \a timestep has already been set, this overwrites it.
*/
void VariantLinear::setPoint(unsigned int timestep, double val)
    {
    m_values[timestep] = val;
    }

/*! \param timestep Timestep to get the value at
    \return Interpolated value
*/
double VariantLinear::getValue(unsigned int timestep)
    {
    // first transform the timestep by the offset
    if (timestep < m_offset)
        timestep = 0;
    else
        timestep -= m_offset;

    // handle the degenerate case that the variant is empty
    if (m_values.empty())
        {
        throw runtime_error("Error: No points specified to VariantLinear");
        }

    // handle the degernate case that there is only one value in the variant
    if (m_values.size() == 1)
        return m_values.begin()->second;

    // handle beginning case
    if (timestep < m_values.begin()->first)
        return m_values.begin()->second;

    // handle end case
    map<unsigned int, double>::iterator last = m_values.end();
    --last;
    if (timestep >= last->first)
        return last->second;

    // handle middle case
    // check to see if the cache is still correct
    bool cache_ok = m_a != m_values.end() && m_b != m_values.end() && timestep >= m_a->first && timestep < m_b->first;
    if (!cache_ok)
        {
        // reload the cached iterators
        m_a = m_values.upper_bound(timestep);

        m_b = m_a;
        --m_a;
        assert(m_a != m_values.end());
        assert(m_b != m_values.end());
        }

    // interpolate
    unsigned int ta = m_a->first;
    unsigned int tb = m_b->first;
    assert(tb > ta);

    double va = m_a->second;
    double vb = m_b->second;

    assert(timestep >= ta && timestep < tb);
    double f = double((timestep - ta)) / double((tb - ta));
    return (1.0 - f) * va + f * vb;
    }

void export_Variant()
    {
    class_<Variant, boost::shared_ptr<Variant> >("Variant", init< >())
    .def("getValue", &Variant::getValue)
    .def("setOffset", &Variant::setOffset);

    class_<VariantConst, boost::shared_ptr<VariantConst>, bases<Variant> >("VariantConst", init< double >());

    class_<VariantLinear, boost::shared_ptr<VariantLinear>, bases<Variant> >("VariantLinear", init< >())
    .def("setPoint", &VariantLinear::setPoint);
    }
