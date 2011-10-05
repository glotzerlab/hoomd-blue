/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// Maintainer: joaander

/*! \file Variant.cc
    \brief Defines Variant and related classes
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

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
        cerr << endl << "***Error! No points specified to VariantLinear" << endl << endl;
        throw runtime_error("Error getting variant value");
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

#ifdef WIN32
#pragma warning( pop )
#endif

