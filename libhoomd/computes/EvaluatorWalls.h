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

/*! \file EvaluatorWalls.h
    \brief Executes an external field potential of several evaluator types for each wall in the system.
 */

#ifndef __EVALUATOR_WALLS_H__
#define __EVALUATOR_WALLS_H__

#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include <boost/bind.hpp>
#include "HOOMDMath.h"
#include "VectorMath.h"
#include "WallData.h"

#undef DEVICE
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

// sets the max numbers for each wall geometry type
const unsigned int MAX_N_SWALLS=20;
const unsigned int MAX_N_CWALLS=20;
const unsigned int MAX_N_PWALLS=60;

struct wall_type{
    SphereWall       Spheres[MAX_N_SWALLS];
    CylinderWall     Cylinders[MAX_N_CWALLS];
    PlaneWall        Planes[MAX_N_PWALLS];
    unsigned int     numSpheres;
    unsigned int     numCylinders;
    unsigned int     numPlanes;
} __attribute__((aligned(8)));   // align on a 8 byte boundary so the GPU kernel can load it in 4 bytes at a time

//! Applys a wall force from all walls in the field parameter
/*! \ingroup computes
*/
template<class evaluator>
class EvaluatorWalls
    {
    public:
        typedef struct
            {
            typename evaluator::param_type params;
            Scalar rcutsq;
            Scalar rextrap;
            } param_type;

        typedef wall_type field_type;

        //! Constructs the external wall potential evaluator
        DEVICE EvaluatorWalls(Scalar3 pos, const BoxDim& box, const param_type& p, const field_type& f) : m_pos(pos), m_field(f), m_params(p)
            {
            }

        //! Test if evaluator needs Diameter
        DEVICE static bool needsDiameter()
            {
            return evaluator::needsDiameter();
            }

        //! Accept the optional diameter value
        /*! \param di Diameter of particle i
        */
        DEVICE void setDiameter(Scalar diameter)
            {
            di = diameter;
            }

        //! Charges not supported by walls evals
        DEVICE static bool needsCharge()
            {
            return evaluator::needsCharge();
            }

        //! Declares additional virial cotribututions are needed for the external field
        DEVICE static bool requestFieldVirialTerm()
            {
            return false; //volume change dependence is not currently defined
            }

        //! Accept the optional charge value
        /*! \param qi Charge of particle i
        Walls charge currently assigns a charge of 0 to the walls. It is however unused by implemented potentials.
        */
        DEVICE void setCharge(Scalar charge)
            {
            qi = charge;
            }

        DEVICE inline void callEvaluator(Scalar3& F, Scalar& energy, const vec3<Scalar> drv)
            {
            Scalar3 dr = -vec_to_scalar3(drv);
            Scalar rsq = dot(dr, dr);

            // compute the force and potential energy
            Scalar force_divr = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);
            evaluator eval(rsq, m_params.rcutsq, m_params.params);
            if (evaluator::needsDiameter())
                eval.setDiameter(di, Scalar(0.0));
            if (evaluator::needsCharge())
                eval.setCharge(qi, Scalar(0.0));

            bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, true);

            if (evaluated)
                {
                // add the force and potential energy to the particle i
                F += dr*force_divr;
                energy = pair_eng; // removing half since the other "particle" won't be represented * Scalar(0.5);
                }
            }

        DEVICE inline void extrapEvaluator(Scalar3& F, Scalar& energy, const vec3<Scalar> drv, const Scalar rextrapsq, const Scalar r)
            {
            Scalar3 dr = -vec_to_scalar3(drv);
            // compute the force and potential energy
            Scalar force_divr = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);

            evaluator eval(rextrapsq, m_params.rcutsq, m_params.params);
            if (evaluator::needsDiameter())
                eval.setDiameter(di, Scalar(0.0));
            if (evaluator::needsCharge())
                eval.setCharge(qi, Scalar(0.0));

            bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, true);

            if (evaluated)
                {
                // add the force and potential energy to the particle i
                energy = pair_eng + force_divr * m_params.rextrap * r; // removing half since the other "particle" won't be represented * Scalar(0.5);
                force_divr *= m_params.rextrap / r;
                F += dr * force_divr;
                }
            }

        //! Generates force and energy from standard evaluators using wall geometry functions
        DEVICE void evalForceEnergyAndVirial(Scalar3& F, Scalar& energy, Scalar* virial)
            {
            F.x = Scalar(0.0);
            F.y = Scalar(0.0);
            F.z = Scalar(0.0);
            energy = Scalar(0.0);
            // initialize virial
            for (unsigned int i = 0; i < 6; i++)
                virial[i] = Scalar(0.0);

            // convert type as little as possible
            vec3<Scalar> position = vec3<Scalar>(m_pos);
            vec3<Scalar> drv;
            bool inside = false; //keeps compiler from complaining
            if (m_params.rextrap>0.0) //extrapolated mode
                {
                Scalar rextrapsq=m_params.rextrap * m_params.rextrap;
                Scalar rsq;
                for (unsigned int k = 0; k < m_field.numSpheres; k++)
                    {
                    drv = vecPtToWall(m_field.Spheres[k], position, inside);
                    rsq = dot(drv, drv);
                    if (inside && rsq>=rextrapsq)
                        {
                        callEvaluator(F, energy, drv);
                        }
                    else
                        {
                        Scalar r = fast::sqrt(rsq);
                        if (rsq == 0.0)
                            {
                            inside = true; //just in case
                            drv = (position - m_field.Spheres[k].origin) / m_field.Spheres[k].r;
                            }
                        else
                            {
                            drv *= 1/r;
                            }
                        r = (inside) ? m_params.rextrap - r : m_params.rextrap + r;
                        drv *= (inside) ? r : -r;
                        extrapEvaluator(F, energy, drv, rextrapsq, r);
                        }
                    }
                for (unsigned int k = 0; k < m_field.numCylinders; k++)
                    {
                    drv = vecPtToWall(m_field.Cylinders[k], position, inside);
                    rsq = dot(drv, drv);
                    if (inside && rsq>=rextrapsq)
                        {
                        callEvaluator(F, energy, drv);
                        }
                    else
                        {
                        Scalar r = fast::sqrt(rsq);
                        if (rsq == 0.0)
                            {
                            inside = true; //just in case
                            drv = rotate(m_field.Cylinders[k].quatAxisToZRot,position - m_field.Cylinders[k].origin);
                            drv.z = 0.0;
                            drv = rotate(conj(m_field.Cylinders[k].quatAxisToZRot),drv) / m_field.Cylinders[k].r;
                            }
                        else
                            {
                            drv *= 1/r;
                            }
                        r = (inside) ? m_params.rextrap - r : m_params.rextrap + r;
                        drv *= (inside) ? r : -r;
                        extrapEvaluator(F, energy, drv, rextrapsq, r);
                        }
                    }
                for (unsigned int k = 0; k < m_field.numPlanes; k++)
                    {
                    drv = vecPtToWall(m_field.Planes[k], position, inside);
                    rsq = dot(drv, drv);
                    if (inside && rsq>=rextrapsq)
                        {
                        callEvaluator(F, energy, drv);
                        }
                    else
                        {
                        Scalar r = fast::sqrt(rsq);
                        if (rsq == 0.0)
                            {
                            inside = true; //just in case
                            drv = m_field.Planes[k].normal;
                            }
                        else
                            {
                            drv *= 1/r;
                            }
                        r = (inside) ? m_params.rextrap - r : m_params.rextrap + r;
                        drv *= (inside) ? r : -r;
                        extrapEvaluator(F, energy, drv, rextrapsq, r);
                        }
                    }
                }
            else //normal mode
                {
                for (unsigned int k = 0; k < m_field.numSpheres; k++)
                    {
                    drv = vecPtToWall(m_field.Spheres[k], position, inside);
                    if (inside)
                        {
                        callEvaluator(F, energy, drv);
                        }
                    }
                for (unsigned int k = 0; k < m_field.numCylinders; k++)
                    {
                    drv = vecPtToWall(m_field.Cylinders[k], position, inside);
                    if (inside)
                        {
                        callEvaluator(F, energy, drv);
                        }
                    }
                for (unsigned int k = 0; k < m_field.numPlanes; k++)
                    {
                    drv = vecPtToWall(m_field.Planes[k], position, inside);
                    if (inside)
                        {
                        callEvaluator(F, energy, drv);
                        }
                    }
                }

            // evaluate virial
            virial[0] = F.x*m_pos.x;
            virial[1] = F.x*m_pos.y;
            virial[2] = F.x*m_pos.z;
            virial[3] = F.y*m_pos.y;
            virial[4] = F.y*m_pos.z;
            virial[5] = F.z*m_pos.z;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("walls_") + evaluator::getName();
            }
        #endif

    protected:
        Scalar3     m_pos;                //!< particle position
        const field_type&  m_field;       //!< contains all information about the walls.
        param_type  m_params;
        Scalar      di;
        Scalar      qi;
    };

template < class evaluator >
typename EvaluatorWalls<evaluator>::param_type make_wall_params(typename evaluator::param_type p, Scalar rcutsq, Scalar rextrap)
    {
    typename EvaluatorWalls<evaluator>::param_type params;
    params.params = p;
    params.rcutsq = rcutsq;
    params.rextrap = rextrap;
    return params;
    }
#endif //__EVALUATOR__WALLS_H__
#ifndef NVCC

//! Exports helper function for parameters based on standard evaluators
template< class evaluator >
void export_wall_params_helpers()
    {
    class_<typename EvaluatorWalls<evaluator>::param_type , boost::shared_ptr<typename EvaluatorWalls<evaluator>::param_type> >((EvaluatorWalls<evaluator>::getName()+"_params").c_str(), init<>())
        .def_readwrite("params", &EvaluatorWalls<evaluator>::param_type::params)
        .def_readwrite("rextrap", &EvaluatorWalls<evaluator>::param_type::rextrap)
        .def_readwrite("rcutsq", &EvaluatorWalls<evaluator>::param_type::rcutsq)
        ;
    def(std::string("make_"+EvaluatorWalls<evaluator>::getName()+"_params").c_str(), &make_wall_params<evaluator>);
    }

//! Combines exports of evaluators and parameter helper functions
template < class evaluator >
void export_PotentialExternalWall(const std::string& name)
    {
    export_PotentialExternal< PotentialExternal<EvaluatorWalls<evaluator> > >(name);
    export_wall_params_helpers<evaluator>();
    }

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
#endif
