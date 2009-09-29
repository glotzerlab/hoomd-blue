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

// $Id: BD_NVTUpdaterGPU.h 1045 2008-07-07 21:35:57Z phillicl $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/updaters_gpu/BD_NVTUpdaterGPU.h $
// Maintainer: phillicl

/*! \file BD_NVTUpdaterGPU.h
    \brief Declares the BD_NVTUpdaterGPU class
*/

#include "BD_NVTUpdater.h"

#include <boost/shared_ptr.hpp>

#ifndef __BD_NVTUPDATER_GPU_H__
#define __BD_NVTUPDATER_GPU_H__

//! Brownian dynamics integration of particles
/*! \ingroup updaters
    See BD_NVTUpdater for details. This class implements the same calculations on the GPU.
*/
class BD_NVTUpdaterGPU : public BD_NVTUpdater
    {
    public:
        //! Constructor
        BD_NVTUpdaterGPU(boost::shared_ptr<SystemDefinition> sysdef, Scalar deltaT, boost::shared_ptr<Variant> Temp, unsigned int seed, bool use_diam);
        
        //! Take one timestep forward
        virtual void update(unsigned int timestep);
    };

//! Exports the BD_NVTUpdaterGPU class to python
void export_BD_NVTUpdaterGPU();

#endif
