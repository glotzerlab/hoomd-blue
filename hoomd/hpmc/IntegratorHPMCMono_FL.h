// inclusion guard
#ifndef _INTEGRATOR_HPMC_MONO_FL_H_
#define _INTEGRATOR_HPMC_MONO_FL_H_

/*! \file IntegratorHPMCMono_FL.h
    \brief Declaration of IntegratorHPMC
*/


#include "hoomd/Integrator.h"
#include <boost/python.hpp>
#include "IntegratorHPMC.h"
#include "IntegratorHPMCMono.h"
#include "Moves.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

namespace hpmc
{

//! HPMC on systems of shapes
/*! Implement hard particle monte carlo with a Frenkel Ladd potential for a single type of shape on the CPU.

    If the simulation box is too small relative to the maximum particle dimension,
    the HOOMD cell list cannot be used and a slower code path is followed. A HOOMD
    warning and loggable quantities are available to detect and analyze this condition
    in case it occurs as a result of a pathological simulation, but the alternate code
    path makes possible single unit cell and densest packing experiments not possible
    in the GPU code path.

    \ingroup hpmc_integrators
*/
template < class Shape >
class IntegratorHPMCMono_FL : public IntegratorHPMCMono<Shape>
    {
    public:
        //! Param type from the shape
        typedef typename Shape::param_type param_type;

        //! Constructor
        IntegratorHPMCMono_FL(boost::shared_ptr<SystemDefinition> sysdef,
                              unsigned int seed);

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        //! Calculate the FL energy
        void calc_fl_energy();

        //! Set the FL spring parameters, and reset updater state
        void setLnGamma(Scalar ln_gamma){m_ln_gamma = ln_gamma;}
        //! Set the FL parameter q_factor
        void setQFactor(Scalar q_factor){m_q_factor = q_factor;}
        //! Set the FL parameter drift_period
        void setDriftPeriod(unsigned int period){m_drift_period=period;}
        //! Set the FL parameter r0
        void setR0(boost::python::list r0);
        //! Set the FL parameter q0
        void setQ0(boost::python::list q0);

        //! Reset the FL energy averaging
        void resetU_FL(){U_FL_sum = U_FL_y = U_FL_t = U_FL_c = 0.0f;}
        //! Reset the FL energy averaging
        void resetUsq_FL(){ Usq_FL_sum = Usq_FL_y = Usq_FL_t = Usq_FL_c = 0.0f;}
        //! Reset the FL energy averaging
        void resetState(unsigned int timestep);
        //! Remove drive
        void removeDrift();
        //! Get the FL logged quantities
        Scalar getLogValue(const std::string& quantity, unsigned int timestep);
        //! Get the FL logged quantities
        std::vector< std::string > getProvidedLogQuantities();

    private:

        //! GPU Arrays for FL Einsten xtals
        GPUArray<Scalar3> m_r0;
        //! GPU Arrays for FL Einsten xtals
        GPUArray<Scalar4> m_q0;

        //FL spring constants
        Scalar m_ln_gamma;    //!< Natural log of FL spring constant
        Scalar m_q_factor;    //!< Natural log of FL spring constant

        Scalar U_FL_sum;      //!< State for logging U_FL
        Scalar U_FL_y;        //!< State for logging U_FL
        Scalar U_FL_t;        //!< State for logging U_FL
        Scalar U_FL_c;        //!< State for logging U_FL
        Scalar Usq_FL_sum;    //!< State for logging U_FL
        Scalar Usq_FL_y;      //!< State for logging U_FL
        Scalar Usq_FL_t;      //!< State for logging U_FL
        Scalar Usq_FL_c;      //!< State for logging U_FL
        unsigned int last_U_FL_timestep; //!< State for logging U_FL
        unsigned int m_drift_period;     //!< drift period
    };

/*! \param sysdef HOOMD system definition
    \param  cl HOOMD cell list
    \param  seed RNG seed
*/
template <class Shape>
IntegratorHPMCMono_FL<Shape>::IntegratorHPMCMono_FL(boost::shared_ptr<SystemDefinition> sysdef,
                                                   unsigned int seed)
            : IntegratorHPMCMono<Shape>(sysdef, seed),
              m_drift_period(1000)
    {

    // allcate arrays for the Einstein xtal coordinates
    GPUArray<Scalar3> r0(this->m_pdata->getN(), this->m_exec_conf);
    m_r0.swap(r0);
    GPUArray<Scalar4> q0(this->m_pdata->getN(), this->m_exec_conf);
    m_q0.swap(q0);

    resetU_FL();
    resetUsq_FL();
    last_U_FL_timestep=0;
    }

/*! Resets the running averages kept by the integrator
 * \param timestep current timestep
*/
template <class Shape>
void IntegratorHPMCMono_FL<Shape>::resetState(unsigned int timestep)
    {
    resetU_FL();
    resetUsq_FL();
    last_U_FL_timestep=timestep;
    }

/*! Initialize einstein coords
 * \param r0 python list of Einstein crystal translation coordinates
*/
template <class Shape>
void IntegratorHPMCMono_FL<Shape>::setR0(boost::python::list r0)
    {
    // validate input type and rank
    boost::python::ssize_t n = boost::python::len(r0);
    ArrayHandle<Scalar3> h_r0(m_r0, access_location::host, access_mode::read);
    for ( boost::python::ssize_t i=0; i<n; i++)
      {
      boost::python::tuple r0_tuple = boost::python::extract<boost::python::tuple >(r0[i]);
      h_r0.data[i].x=boost::python::extract<Scalar>(r0_tuple[0]);
      h_r0.data[i].y=boost::python::extract<Scalar>(r0_tuple[1]);
      h_r0.data[i].z=boost::python::extract<Scalar>(r0_tuple[2]);
      }
    }

/*! Initialize Einstein coords
 * \param q0 python list of Einstein crystal orientation coordinates
*/
template <class Shape>
void IntegratorHPMCMono_FL<Shape>::setQ0(boost::python::list q0)
    {
    // validate input type and rank
    boost::python::ssize_t n = boost::python::len(q0);
    ArrayHandle<Scalar4> h_q0(m_q0, access_location::host, access_mode::read);
    for ( boost::python::ssize_t i=0; i<n; i++)
      {
      boost::python::tuple q0_tuple = boost::python::extract<boost::python::tuple >(q0[i]);
      h_q0.data[i].x=boost::python::extract<Scalar>(q0_tuple[0]);
      h_q0.data[i].y=boost::python::extract<Scalar>(q0_tuple[1]);
      h_q0.data[i].z=boost::python::extract<Scalar>(q0_tuple[2]);
      h_q0.data[i].w=boost::python::extract<Scalar>(q0_tuple[3]);
      }
    }


template <class Shape>
std::vector< std::string > IntegratorHPMCMono_FL<Shape>::getProvidedLogQuantities()
    {
    // start with the integrator provided quantities
    std::vector< std::string > result = IntegratorHPMCMono<Shape>::getProvidedLogQuantities();
    // then add ours
    result.push_back("hpmc_U_FL");
    result.push_back("hpmc_sigma_U_FL");
    result.push_back("hpmc_U_FL_interval");
    result.push_back("hpmc_ln_gamma");
    result.push_back("hpmc_q_factor");

    return result;
    }

/*! Get FL specific quantities
    \param quantity quantity to be logged
    \param timestep current timestep
    \returns the log value
*/
template <class Shape>
Scalar IntegratorHPMCMono_FL<Shape>::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == "hpmc_ln_gamma")
        return m_ln_gamma;
    else if (quantity == "hpmc_q_factor")
        return m_q_factor;
    else if (quantity == "hpmc_U_FL")
        {
        if (timestep==last_U_FL_timestep)
            return 0.0f;
        return U_FL_sum/(this->m_nselect*double(timestep-last_U_FL_timestep));
        }
    else if (quantity == "hpmc_sigma_U_FL")
        {
        if (timestep==last_U_FL_timestep)
            return 0.0f;
        Scalar U_sq_FL =  Usq_FL_sum/(this->m_nselect*double(timestep-last_U_FL_timestep));
        Scalar U_FL =  U_FL_sum/(this->m_nselect*double(timestep-last_U_FL_timestep));
        return sqrt(U_sq_FL-U_FL*U_FL);
        }
    else if (quantity == "hpmc_U_FL_interval")
        {
        return (timestep-last_U_FL_timestep);
        }
    return IntegratorHPMCMono<Shape>::getLogValue(quantity,timestep) ;
    }

/*! Do a monte carlo sweep, and reset any COM drift at the drift_period
 * \param timestep current timestep
*/
template <class Shape>
void IntegratorHPMCMono_FL<Shape>::update(unsigned int timestep)
    {
    #ifdef ENABLE_MPI
    if (this->m_comm)
        {
        this->m_exec_conf->msg->error() << "FL updates do not work in MPI" << std::endl;
        throw std::runtime_error("Error updating with FL on");
        }
    #endif
    this->m_exec_conf->msg->notice(10) << "HPMCMono_FL update: " << timestep << std::endl;
    IntegratorHPMC::update(timestep);

    //reset the COM drift
    if (timestep%m_drift_period==0)
        {
        removeDrift();
        this->m_exec_conf->msg->notice(10) << "HPMC drift removed: " << timestep << std::endl;
        }

    // Shuffle the order of particles for this step
    this->m_update_order.resize(this->m_pdata->getN());
    this->m_update_order.shuffle(timestep);

    // get needed data
    ArrayHandle<hpmc_counters_t> h_counters(this->m_count_total, access_location::host, access_mode::readwrite);
    hpmc_counters_t& counters = h_counters.data[0];
    unsigned int ndim = this->m_sysdef->getNDimensions();
    const BoxDim& box = this->m_pdata->getBox();

    // build the AABB Tree
    this->buildAABBTree();
    // limit m_d entries so that particles cannot possibly wander more than one box image in one time step
    this->limitMoveDistances();
    // update the image list
    this->updateImageList();

    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "HPMC update");

    // loop over local particles nselect times
    for (unsigned int i_nselect = 0; i_nselect < this->m_nselect; i_nselect++)
        {
        // access particle data and system box
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::readwrite);

        // access parameters
        ArrayHandle<param_type> h_params(this->m_params, access_location::host, access_mode::read);

        //access move sizes
        ArrayHandle<Scalar> h_d(this->m_d, access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_a(this->m_a, access_location::host, access_mode::read);

        //calc_fl_energy();

        ArrayHandle<Scalar3> h_r0(m_r0, access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_q0(m_q0, access_location::host, access_mode::read);

        Scalar U_FL_tmp(0.0f);
        Scalar gamma = fast::exp(m_ln_gamma);
        // loop through N particles in a shuffled order
        for (unsigned int cur_particle = 0; cur_particle < this->m_pdata->getN(); cur_particle++)
            {
            unsigned int i = this->m_update_order[cur_particle];

            // read in the current position and orientation
            Scalar4 postype_i = h_postype.data[i];
            Scalar4 orientation_i = h_orientation.data[i];
            unsigned int tag_i = h_tag.data[i];

            vec3<Scalar> pos_i = vec3<Scalar>(postype_i);
            vec3<Scalar> new_pos_i = vec3<Scalar>(postype_i);
            quat<Scalar> quat_i = quat<Scalar>(orientation_i);
            quat<Scalar> new_quat_i = quat<Scalar>(orientation_i);
            Shape shape_i(new_quat_i, h_params.data[__scalar_as_int(postype_i.w)]);

            //Grab the FL positions and declare some FL variables
            vec3<Scalar> r0 = vec3<Scalar>(h_r0.data[tag_i]);
            quat<Scalar> q0 = quat<Scalar>(h_q0.data[tag_i]);
            Scalar new_U(0.0f);
            Scalar old_U(0.0f);
            Scalar boltz(0.0f);

            // make a trial move for i
            unsigned int typ_i = __scalar_as_int(postype_i.w);
            Saru rng_i(i, this->m_seed +this->m_exec_conf->getRank()*this->m_nselect + i_nselect, timestep);
            Scalar move_type_select = rng_i.u32() & 0xffff;
            bool move_type_translate = !shape_i.hasOrientation() || (move_type_select < this->m_move_ratio);

            if (move_type_translate)
                {
                move_translate(new_pos_i, rng_i,h_d.data[typ_i], ndim);
                }
            else
                {
                move_rotate(new_quat_i, rng_i,h_a.data[typ_i], ndim);

                // update Shape
                shape_i.orientation = new_quat_i;
                }

            // calc boltzmann factor from springs
            vec3<Scalar> dr = vec3<Scalar>(box.minImage(vec_to_scalar3(r0 - new_pos_i)));
            quat<Scalar> dq = q0 - new_quat_i;
            new_U = gamma*(dot(dr,dr)+m_q_factor*norm2(dq));

            dr = vec3<Scalar>(box.minImage(vec_to_scalar3(r0 - pos_i)));
            dq = q0 - quat_i;
            old_U = gamma*(dot(dr,dr)+m_q_factor*norm2(dq));

            U_FL_tmp += old_U;

            boltz= fast::exp(old_U-new_U);

            bool overlap = false;
            if(rng_i.s(Scalar(0.0),Scalar(1.0)) < detail::min(Scalar(1.0),boltz))
                overlap = false;
            else
                overlap = true; // treat fl reject as an "overlap"

            // check for overlaps with neighboring particle's positions

            detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

            // All image boxes (including the primary)
            const unsigned int n_images = this->m_image_list.size();
            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_i_image = new_pos_i + this->m_image_list[cur_image];
                detail::AABB aabb = aabb_i_local;
                aabb.translate(pos_i_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < this->m_aabb_tree.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(this->m_aabb_tree.getNodeAABB(cur_node_idx), aabb))
                        {
                        if (this->m_aabb_tree.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < this->m_aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = this->m_aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                Scalar4 postype_j;
                                Scalar4 orientation_j;

                                // handle j==i situations
                                if ( j != i )
                                    {
                                    // load the position and orientation of the j particle
                                    postype_j = h_postype.data[j];
                                    orientation_j = h_orientation.data[j];
                                    }
                                else
                                    {
                                    if (cur_image == 0)
                                        {
                                        // in the first image, skip i == j
                                        continue;
                                        }
                                    else
                                        {
                                        // If this is particle i and we are in an outside image, use the translated position and orientation
                                        postype_j = make_scalar4(new_pos_i.x, new_pos_i.y, new_pos_i.z, postype_i.w);
                                        orientation_j = quat_to_scalar4(new_quat_i);
                                        }
                                    }

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i_image;

                                Shape shape_j(quat<Scalar>(orientation_j), h_params.data[__scalar_as_int(postype_j.w)]);

                                counters.overlap_checks++;
                                if (!(shape_i.ignoreOverlaps() && shape_j.ignoreOverlaps())
                                    && check_circumsphere_overlap(r_ij, shape_i, shape_j)
                                    && test_overlap(r_ij, shape_i, shape_j, counters.overlap_err_count))
                                    {
                                    overlap = true;
                                    break;
                                    }
                                }
                            }
                        }
                    else
                        {
                        // skip ahead
                        cur_node_idx += this->m_aabb_tree.getNodeSkip(cur_node_idx);
                        }

                    if (overlap)
                        break;
                    }  // end loop over AABB nodes

                if (overlap)
                    break;
                } // end loop over images

            // if the move is accepted
            if (!overlap)
                {
                // increment accept counter and assign new position
                if (!shape_i.ignoreStatistics())
                  {
                  if (move_type_translate)
                      counters.translate_accept_count++;
                  else
                      counters.rotate_accept_count++;
                  }
                // update the position of the particle in the tree for future updates
                detail::AABB aabb = aabb_i_local;
                aabb.translate(new_pos_i);
                this->m_aabb_tree.update(i, aabb);

                // update position of particle
                h_postype.data[i] = make_scalar4(new_pos_i.x,new_pos_i.y,new_pos_i.z,postype_i.w);

                if (shape_i.hasOrientation())
                    {
                    // update orientation
                    h_orientation.data[i] = quat_to_scalar4(new_quat_i);
                    }
                }

             else
                {
                if (!shape_i.ignoreStatistics())
                    {
                    // increment reject counter
                    if (move_type_translate)
                        counters.translate_reject_count++;
                    else
                        counters.rotate_reject_count++;
                    }
                }
            } // end loop over all particles

        //Add this steps energy to the total energy
        //Use Kahan Sum to avoid round off
        //Update to double precision in hybrid branch?
        U_FL_tmp=U_FL_tmp/this->m_pdata->getN();
        U_FL_y    = U_FL_tmp - U_FL_c;
        U_FL_t    = U_FL_sum + U_FL_y;
        U_FL_c    = (U_FL_t-U_FL_sum) - U_FL_y;
        U_FL_sum  = U_FL_t;

        U_FL_tmp*=U_FL_tmp;
        Usq_FL_y    = U_FL_tmp - Usq_FL_c;
        Usq_FL_t    = Usq_FL_sum + Usq_FL_y;
        Usq_FL_c    = (Usq_FL_t-Usq_FL_sum) - Usq_FL_y;
        Usq_FL_sum  = Usq_FL_t;
        } // end loop over nselect

        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<int3> h_image(this->m_pdata->getImages(), access_location::host, access_mode::readwrite);

        // wrap particles back into box
        for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
            {
            box.wrap(h_postype.data[i], h_image.data[i]);
            }
        }

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

    // all particle have been moved, the aabb tree is now invalid
    this->m_aabb_tree_invalid = true;
    }


/*! Remove center of mass drift of the system
 */
template<class Shape>
void IntegratorHPMCMono_FL<Shape>::removeDrift()
    {
    ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_r0(m_r0, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(this->m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(this->m_pdata->getImages(), access_location::host, access_mode::readwrite);
    const BoxDim& box = this->m_pdata->getBox();

    vec3<Scalar> rshift;
    rshift.x=rshift.y=rshift.z=0.0f;

    for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
        {
        unsigned int tag_i = h_tag.data[i];
        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        vec3<Scalar> dr = vec3<Scalar>(postype_i) - vec3<Scalar>(h_r0.data[tag_i]);
        rshift += vec3<Scalar>(box.minImage(vec_to_scalar3(dr)));
        }

    rshift/=Scalar(this->m_pdata->getN());

    for (unsigned int i = 0; i < this->m_pdata->getN(); i++)
        {
        // read in the current position and orientation
        Scalar4 postype_i = h_postype.data[i];
        vec3<Scalar> r_i = vec3<Scalar>(postype_i);
        h_postype.data[i] = vec_to_scalar4(r_i - rshift, postype_i.w);
        box.wrap(h_postype.data[i], h_image.data[i]);
        }

    this->m_aabb_tree_invalid = true;
    }

//! Export the IntegratorHPMCMono_FL class to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of IntegratorHPMCMono_FL<Shape> will be exported
*/
template < class Shape > void export_IntegratorHPMCMono_FL(const std::string& name)
    {
    boost::python::class_< IntegratorHPMCMono_FL<Shape>, boost::shared_ptr< IntegratorHPMCMono_FL<Shape> >, boost::python::bases<IntegratorHPMCMono<Shape>, IntegratorHPMC >, boost::noncopyable >
      (name.c_str(), boost::python::init< boost::shared_ptr<SystemDefinition>, unsigned int >())
          .def("setParam", &IntegratorHPMCMono_FL<Shape>::setParam)
          .def("setLnGamma", &IntegratorHPMCMono_FL<Shape>::setLnGamma)
          .def("setQFactor", &IntegratorHPMCMono_FL<Shape>::setQFactor)
          .def("setR0", &IntegratorHPMCMono_FL<Shape>::setR0)
          .def("setQ0", &IntegratorHPMCMono_FL<Shape>::setQ0)
          .def("setDriftPeriod", &IntegratorHPMCMono_FL<Shape>::setDriftPeriod)
          .def("resetState", &IntegratorHPMCMono_FL<Shape>::resetState)
          .def("removeDrift", &IntegratorHPMCMono_FL<Shape>::removeDrift)
          ;
    }

} // end namespace hpmc

#endif // _INTEGRATOR_HPMC_MONO_H_
