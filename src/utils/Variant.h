/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$

/*! \file Variant.h
	\brief Declares the Variant and related classes
*/

#ifndef __VARIANT_H__
#define __VARIANT_H__

#include <map>

//! Base type for time varying quantities
/*! Virtual base class for variables specified to vary over time.
	\ingroup utils
*/
class Variant
	{
	public:
		//! Virtual destructor
		virtual ~Variant() { }
		//! Gets the value at a given time step (just returns 0)
		virtual double getValue(unsigned int timestep) { return 0.0; }
	};

//! Constant variant
/*! Specifies a value that is constant over all time */
class VariantConst : public Variant
	{
	public:
		//! Constructor
		VariantConst(double val) : m_val(val)
			{
			}
		//! Gets the value at a given time step
		virtual double getValue(unsigned int timestep)
				{
				return m_val;
				}
	
	private:
		double m_val;		//!< The value
	};

//! Linearly interpolated variant
/*! This variant is given a series of timestep,value pairs. The value at each time step is
	calculated via linear interpolation between the two nearest points. When the timestep 
	requested is before or after the end of the specified points, the value at the beginning
	(or end) is returned.
*/
class VariantLinear : public Variant
	{
	public:
		//! Constructs an empty variant
		VariantLinear();
		//! Gets the value at a given time step
		virtual double getValue(unsigned int timestep);
		//! Sets a point in the interpolation
		void setPoint(unsigned int timestep, double val);
		
	private:
		std::map<unsigned int, double> m_values;	//!< Values to interpoloate
		std::map<unsigned int, double>::iterator	m_a,	//!< First point in the pair to interpolate
													m_b;	//!< Second point in the pair to inerpolate
	};
		
//! Exports Variant* classes to python
void export_Variant();
	
#endif
