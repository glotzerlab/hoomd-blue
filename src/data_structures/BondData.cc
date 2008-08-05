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

#include "BondData.h"
#include "ParticleData.h"

#include <stdexcept>

using namespace std;

/*! \file BondData.cc
 	\brief Contains definitions for BondData.
 */

/*! \param pdata ParticleData these bonds refer into
	\param n_bond_types Number of bond types in the list
*/
BondData(boost::shared_ptr<ParticleData> pdata, unsigned int n_bond_types) : m_n_bond_types(n_bond_types), m_pdata(pdata)
	{
	assert(pdata);
	}

/*! \post A bond between particles specified in \a bond is created. 
	
	\note Each bond should only be specified once! There are no checks to prevent one from being 
	specified more than once, and doing so would result in twice the force and twice the energy.
	For a bond between \c i and \c j, only call \c addBond(Bond(type,i,j)). Do NOT additionally call 
	\c addBond(Bond(type,j,i)). The first call is sufficient to include the forces on both particle 
	\c i and \c j.
	
	\note If a bond is added with \c type=49, then there must be at least 50 bond types (0-49) total,
	even if not all values are used. So bonds should be added with small contiguous types.
	\param bond The Bond to add to the list
 */	
void BondData::addBond(const Bond& bond)
	{
	// check for some silly errors a user could make 	
	if (bond.a >= m_pdata->getN() || bond.b >= m_pdata->getN())
		{
		cerr << endl << "***Error! Particle tag out of bounds when attempting to add bond: " << bond.a << "," << bond.b << endl << endl;
		throw runtime_error("Error adding bond");
		}
		
	if (bond.a == bond.b)
		{
		cerr << endl << "***Error! Particle cannot be bonded to itself! " << bond.a << "," << bond.b << endl << endl;
		throw runtime_error("Error adding bond");
		}
	
	// check that the type is within bouds
	if (bond.type+1 > m_n_bond_types)
		{
		cerr << endl << "***Error! Invalid bond type! " << bond.type << ", the number of types is " << m_n_bond_types << endl << endl;
		throw runtime_error("Error adding bond");
		}

	m_bonds.push_back(bond);
	}
	
/*! \param bond_type_mapping Mapping array to set
	\c bond_type_mapping[type] should be set to the name of the bond type with index \c type.
	The vector \b must have \c n_bond_types elements in it.
*/
void BondData::setBondTypeMapping(const std::vector<std::string>& bond_type_mapping)
	{
	assert(m_bond_type_mapping.size() == m_n_bond_types);
	m_bond_type_mapping = bond_type_mapping;
	}
	

/*! \param name Type name to get the index of
	\return Type index of the corresponding type name
	\note Throws an exception if the type name is not found
*/
unsigned int BondData::getTypeByName(const std::string &name)
	{
	// search for the name
	for (unsigned int i = 0; i < m_bond_type_mapping.size(); i++)
		{
		if (m_bond_type_mapping[i] == name)
			return i;
		}
		
	cerr << endl << "***Error! Bond type " << name << " not found!" << endl;
	throw runtime_error("Error mapping type name");	
	return 0;
	}
		
/*! \param type Type index to get the name of
	\returns Type name of the requested type
	\note Type indices must range from 0 to getNTypes or this method throws an exception.
*/
std::string BondData::getNameByType(unsigned int type)
	{
	// check for an invalid request
	if (type >= m_n_bond_types)
		{
		cerr << endl << "***Error! Requesting type name for non-existant type " << type << endl << endl;
		throw runtime_error("Error mapping type name");
		}
		
	// return the name
	return m_type_mapping[type];
	}
