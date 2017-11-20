#ifndef _PATCH_ENERGY_JIT_H_
#define _PATCH_ENERGY_JIT_H_

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/hpmc/IntegratorHPMC.h"

#include "OrcLazyJIT.h"

#define PATCH_ENERGY_LOG_NAME           "patch_energy"
#define PATCH_ENERGY_RCUT               "patch_energy_rcut"

//! Evaluate patch energies via runtime generated code
/*! This class enables the widest possible use-cases of patch energies in HPMC with low energy barriers for users to add
    custom interactions that execute with high performance. It provides a generic interface for returning the energy of
    interaction between a pair of particles. The actual computation is performed by code that is loaded and compiled at
    run time using LLVM.

    The user provides LLVM IR code containing a function 'eval' with the defined function signature. On construction,
    this class uses the LLVM library to compile that IR down to machine code and obtain a function pointer to call.

    This is the first use of LLVM in HOOMD and it is experimental. As additional areas are identified as
    useful applications of LLVM, we will want to factor out some of the comment elements of this code
    into a generic LLVM module class. (i.e. handle broadcasting the string and compiling it in one place,
    with specific implementations requesting the function pointers they need).

    LLVM execution is managed with the OrcLazyJIT class in m_JIT. On construction, the LLVM module is loaded and
    compiled. OrcLazyJIT handles construction of C++ static members, etc.... When m_JIT is deleted, all of the compiled
    code and memory used in the module is deleted. OrcLazyJIT takes care of destructing C++ static members inside the
    module.

    LLVM JIT is capable of calling any function in the hosts address space. PatchEnergyJIT does not take advantage of
    that, limiting the user to a very specific API for computing the energy between a pair of particles.
*/
class PatchEnergyJIT : public hpmc::PatchEnergy
    {
    public:
        //! Constructor
        PatchEnergyJIT(std::shared_ptr<ExecutionConfiguration> exec_conf, const std::string& fname, Scalar r_cut);

        //! Get the maximum r_ij radius beyond which energies are always 0
        Scalar getRCut()
            {
            return m_r_cut;
            }

        //! evaluate the energy of the patch interaction
        /*! \param r_ij Vector pointing from particle i to j
            \param type_i Integer type index of particle i
            \param q_i Orientation quaternion of particle i
            \param type_j Integer type index of particle j
            \param q_j Orientation quaternion of particle j

            \returns Energy of the patch interaction.
        */
        float energy(const vec3<float>& r_ij, unsigned int type_i, const quat<float>& q_i, unsigned int type_j, const quat<float>& q_j)
            {
            return m_eval(r_ij, type_i, q_i, type_j, q_j);
            }

        double computePatchEnergy(const ArrayHandle<Scalar4> &positions,const ArrayHandle<Scalar4> &orientations,const BoxDim& box, unsigned int &N)
            {
            double patch_energy = 0.0;
            float r_cut = this->getRCut();
            float r_cut_sq = r_cut*r_cut;

            //const BoxDim& box = m_pdata->getGlobalBox();
            // read in the current position and orientation
            for (unsigned int i = 0; i<N;i++)
                {
                Scalar4 postype_i = positions.data[i];
                Scalar4 orientation_i = orientations.data[i];
                vec3<Scalar> pos_i = vec3<Scalar>(postype_i);
                int typ_i = __scalar_as_int(postype_i.w);
                for (unsigned int j = i+1; j < N; j++)
                    {
                    Scalar4 postype_j = positions.data[j];
                    Scalar4 orientation_j = orientations.data[j];
                    vec3<Scalar> pos_j = vec3<Scalar>(postype_j);
                    vec3<Scalar> dr_ij = pos_j - pos_i;
                    dr_ij = box.minImage(dr_ij);
                    int typ_j = __scalar_as_int(postype_j.w);
                    if (dot(dr_ij,dr_ij) <= r_cut_sq)
                        {
                        patch_energy+=this->energy(dr_ij, typ_i, quat<float>(orientation_i),typ_j, quat<float>(orientation_j));
                        }
                    }
                }
            return patch_energy;
            }

      //   Scalar getLogValue(const std::string& quantity, unsigned int timestep)
      //   {
      //     if ( quantity == PATCH_ENERGY_LOG_NAME )
      //         {
      //           return m_PatchEnergy;
      //         }
      //     else if ( quantity == PATCH_ENERGY_RCUT )
      //         {
      //         return m_r_cut;
      //         }
      //     else
      //         {
      //         //exec_conf->msg->error() << "patch: " << quantity << " is not a valid log quantity" << std::endl;
      //         throw std::runtime_error("Error getting log value");
      //         }
      //   }
      //
      // std::vector< std::string > getProvidedLogQuantities()
      // {
      //   return m_PatchProvidedQuantities;
      // }

    private:
        //! function pointer signature
        typedef float (*EvalFnPtr)(const vec3<float>& r_ij, unsigned int type_i, const quat<float>& q_i, unsigned int type_j, const quat<float>& q_j);
        Scalar m_r_cut;                             //!< Cutoff radius
        std::shared_ptr<llvm::OrcLazyJIT> m_JIT;    //!< JIT execution engine
        EvalFnPtr m_eval;                           //!< Pointer to evaluator function inside the JIT module
        //Scalar m_PatchEnergy;                       //!< patch energy
        //std::vector<std::string>  m_PatchProvidedQuantities; //!< available
    };

//! Exports the PatchEnergyJIT class to python
void export_PatchEnergyJIT(pybind11::module &m);
#endif // _PATCH_ENERGY_JIT_H_
