/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
the University of Michigan All rights reserved.

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

// Maintainer: jproc

/*! \file EvaluatorWalls.cc
    \brief Implements some helper functions for the EvaluatorWalls class
 */

#include "EvaluatorWalls.h"

using namespace boost::python;

//! Helper function for converting python wall group structure to wall_type
wall_type make_wall_field_params(boost::python::object walls, boost::shared_ptr<const ExecutionConfiguration> m_exec_conf)
    {
    wall_type w;
    w.numSpheres = boost::python::len(walls.attr("spheres"));
    w.numCylinders = boost::python::len(walls.attr("cylinders"));
    w.numPlanes = boost::python::len(walls.attr("planes"));

    if (w.numSpheres>MAX_N_SWALLS || w.numCylinders>MAX_N_CWALLS || w.numPlanes>MAX_N_PWALLS)
        {
        m_exec_conf->msg->error() << "A number of walls greater than the maximum number allowed was specified in a wall force." << std::endl;
        throw std::runtime_error("Error loading wall group.");
        }
    else
        {
        for(unsigned int i = 0; i < w.numSpheres; i++)
            {
            Scalar     r = boost::python::extract<Scalar>(walls.attr("spheres")[i].attr("r"));
            Scalar3 origin =boost::python::extract<Scalar3>(walls.attr("spheres")[i].attr("_origin"));
            bool     inside =boost::python::extract<bool>(walls.attr("spheres")[i].attr("inside"));
            w.Spheres[i] = SphereWall(r, origin, inside);
            }
        for(unsigned int i = 0; i < w.numCylinders; i++)
            {
            Scalar     r = boost::python::extract<Scalar>(walls.attr("cylinders")[i].attr("r"));
            Scalar3 origin =boost::python::extract<Scalar3>(walls.attr("cylinders")[i].attr("_origin"));
            Scalar3 axis =boost::python::extract<Scalar3>(walls.attr("cylinders")[i].attr("_axis"));
            bool     inside =boost::python::extract<bool>(walls.attr("cylinders")[i].attr("inside"));
            w.Cylinders[i] = CylinderWall(r, origin, axis, inside);
            }
        for(unsigned int i = 0; i < w.numPlanes; i++)
            {
            Scalar3 origin =boost::python::extract<Scalar3>(walls.attr("planes")[i].attr("_origin"));
            Scalar3 normal =boost::python::extract<Scalar3>(walls.attr("planes")[i].attr("_normal"));
            bool    inside =boost::python::extract<bool>(walls.attr("planes")[i].attr("inside"));
            w.Planes[i] = PlaneWall(origin, normal, inside);
            }
        return w;
        }
    }

//! Exports walls helper function
void export_wall_field_helpers()
    {
    class_< wall_type, boost::shared_ptr<wall_type> >( "wall_type", init<>());
    def("make_wall_field_params", &make_wall_field_params);
    }
