/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: jglaser

/*! \file ParticleSelectorRules.h
    \brief Declares rules for ParticleSelectorRule<T> and ParticleSelectorRuleGPU<T>
*/

#ifndef __PARTICLE_SELECTOR_RULES_H__
#define __PARTICLE_SELECTOR_RULES_H__

#ifdef NVCC
#include <thrust/tuple.h>
#include "ParticleData.cuh"
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#include "ParticleData.h"
#endif

//! Rule to select particles based on their global tag
class GlobalTagRule
    {
    public:
        typedef uint2 param_type; //!< parameter type for storing minimum and maximum tag

        //! Constructor
        //! \param params parameters for this rule
        GlobalTagRule(param_type params)
            :  _tag_min(params.x), _tag_max(params.y)
            {
            }

        //! Method to determine whether a particle is selected
        /*!\param global_tag global particle tag
           \param body body id
           \param type particle type
           \return true if a particle is selected
        */
        HOSTDEVICE inline bool isSelected(unsigned int global_tag, unsigned int body, unsigned int type)
            {
            return (_tag_min <= global_tag && global_tag <= _tag_max);
            }

#ifdef NVCC
        //! Thrust interface for isSelected()
        /*! \param t thrust tuple of tag, body and particle type
         * \return true if particle is selected
         */
        __device__ bool operator() (const thrust::tuple<unsigned int, unsigned int, float4>& t)
            {
            return isSelected(thrust::get<0>(t), thrust::get<1>(t), __float_as_int(thrust::get<2>(t).w));
            }
#endif

    private:
        unsigned int _tag_min;     //! Minimum global tag to select
        unsigned int _tag_max;     //! Maximum global tag to select
    };

//! Rule to select particles based on their type
class TypeRule
    {
    public:
        typedef uint2 param_type; //!< parameter type for storing minimum and maximum particle type

        //! Constructor
        //! \param params parameters for this rule
        TypeRule(param_type params)
            : _type_min(params.x), _type_max(params.y)
            {
            }

        //! Method to determine whether a particle is selected
        /*!\param global_tag global particle tag
           \param body body id
           \param type particle type
           \return true if a particle is selected
        */
        HOSTDEVICE bool isSelected(unsigned int global_tag, unsigned int body, unsigned int type)
            {
            return (_type_min <= type && type <= _type_max);
            }

#ifdef NVCC
        //! Thrust interface for isSelected()
        /*! \param t thrust tuple of tag, body and particle type
         * \return true if particle is selected
         */
        __device__ bool operator() (const thrust::tuple<unsigned int, unsigned int, float4>& t)
            {
            return isSelected(thrust::get<0>(t), thrust::get<1>(t), __float_as_int(thrust::get<2>(t).w));
            }
#endif

    private:
        unsigned int _type_min;     //! Minimum particle tag to select
        unsigned int _type_max;     //! Maximum particle tag to select
    };

//! Rule to select particles that are in rigid bodies
class RigidRule
    {
    public:
        typedef bool param_type;   //!< parameter type

        //! Constructor
        //! \param param parameter for this rule
        RigidRule(bool rigid)
            : _rigid(rigid)
            {
            }

        //! Method to determine whether a particle is selected
        /*!\param global_tag global particle tag
           \param body body id
           \param type particle type
           \return true if a particle is selected
        */
        HOSTDEVICE bool isSelected(unsigned int global_tag, unsigned int body, unsigned int type)
            {
                bool result = false;
                if (_rigid && body != NO_BODY)
                    result = true;
                if (!_rigid && body == NO_BODY)
                    result = true;
                return result;
            }

#ifdef NVCC
        //! Thrust interface for isSelected()
        /*! \param t thrust tuple of tag, body and particle type
         * \return true if particle is selected
         */
        __device__ bool operator() (const thrust::tuple<unsigned int, unsigned int, float4>& t)
            {
            return isSelected(thrust::get<0>(t), thrust::get<1>(t), __float_as_int(thrust::get<2>(t).w));
            }
#endif

    private:
        bool _rigid; //!<true selects particles that are in rigid bodies, false selects particles that are not part of a body
    };
#endif
