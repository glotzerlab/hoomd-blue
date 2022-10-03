// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.



// Used to be called EvaluatorPairIsoModulated


/*
  This class  aplies the directionalEnvelope to the pairEvaluator, turning the isotropic pair potential into an anisotropic potential.
*/


//template <typename isoEval, typename dirEval>
template <typename pairEvaluator, typename directionalEnvelope>
struct PairModulatorParamStruct
{
    typedef typename pairEvaluator::param_type pairParam;
    typedef typename directionalEnvelope::param_type envelopeParam;

    PairModulatorParamStruct()
        {
        }

    PairModulatorParamStruct(pairParam _pairP, envelopeParam _envelP):
        pairP(_pairP), envelP(_envelP)
        {
        }

    pairParam pairP;
    envelopeParam envelP;
};


//! Class to modulate an isotropic pair potential by an envelope to create an anisotropic pair potential.
template <typename pairEvaluator, typename directionalEnvelope>
class PairModulator
{
public:
    typedef PairModulatorParamStruct<pairEvaluator, directionalEnvelope> param_type;

    // Constructs the pair potential evaluator
    /*
      \param _dr Displacement vector pointing from particle rj to ri
      \param _rcutsq Squared distance at which the potential is set to 0
      \param _quat_eye Quaternion of the ith particle
      \param _quat_jay Quaternion of the jth particle
      \param _params Per type pair parameters of the potential
    */
    DEVICE PairModulator( const Scalar3& _dr,,
                          const Scalar4& _quat_eye,
                          const Scalar4& _quat_jay,
                          const Scalar4& _rcutsq,
                          const param_type& _params)
        : dr(_dr),
          rsq(_dr.x*_dr.x + _dr.y*_dr.y + _dr.z*_dr.z),
          rcutsq(_rcutsq),
          pairEval(_dr.x*_dr.x + _dr.y*_dr.y + _dr.z*_dr.z, _rcutsq, _params.pairP),
          envelEval(_dr, _quat_eye, _quat_jay, _rcutsq, _params.envelP)
        { }

    //! If diameter is used
    DEVICE static bool needsDiameter()
        {
            return (pairEvaluator::needsDiameter() || directionalEnvelope::needsDiameter());
        }

    //! Accept the optional diameter values
    /*
      \param di Diameter of particle i
      \param dj Diameter of particle j
    */
    DEVICE void setDiameter(Scalar di, Scalar dj)
        {
            if (pairEvaluator::needsDiameter())
                pairEval.setDiameter(di, dj);
            if (directionalEnvelope::needsDiameter())
                envelEval.setDiameter(di, dj);
        }

    //! Whether pair potential requires charges
    // TODO what does it mean "this function is pure virtual" ?
    DEVICE static bool needsCharge()
        {
            return (pairEvaluator::needsCharge() || directionalEnvelope::needsCharge());
        }

    //! Accept the optional charge values
    /*
      \param qi Charge of particle i
      \param qj Charge of particle j
    */
    DEVICE void setCharge(Scalar qi, Scalar qj)
        {
            if (pairEvaluator::needsCharge())
                pairEval.setCharge(qi, qj);
            if (directionalEnvelope::needsCharge())
                envelEval.setCharge(qi, qj);
        }

    //! Evaluate the force and energy
    /*
      \param force Output parameter to write the computed force.
      \param pair_eng Output parameter to write the computed pair energy.
      \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff.
      \param torque_i The torque exerted on the ith particle.
      \param torque_j The torque exerted on the jth particle.
      \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed 
      in PotentialPair.
      \return True if force and energy are evaluated, false if not because r>rcut.
    */
    DEVICE bool evaluate(Scalar3& force,
                         Scalar& pair_eng,
                         bool energy_shift,
                         Scalar3& torque_i,
                         Scalar3& torque_j)
        {
            // compute pair potential
            Scalar force_divr(Scalar(0));
            if (!pairEval.evalForceAndEnergy(force_divr, pair_eng, energy_shift))
                {
                    return false;
                }

            // compute envelope
            Scalar envelope(Scalar(0));
            envelEval.evaluate(force, envelope, torque_i, torque_j);

            // modulate forces
            // TODO check this math
            force.x = pair_eng*force.x + dr.x*force_divr*envelope;
            force.y = pair_eng*force.y + dr.y*force_divr*envelope;
            force.z = pair_eng*force.z + dr.z*force_divr*envelope;

            // modulate torques
            // TODO check this math
            torque_i.x *= pair_eng;
            torque_i.y *= pair_eng;
            torque_i.z *= pair_eng;

            torque_j.x *= pair_eng;
            torque_j.y *= pair_eng;
            torque_j.z *= pair_eng;

            // modulate pair energy
            pair_eng *= envelope;

            return true;
        }


#ifndef __HIPCC__
    //! Get the name of this potential
    /*
      \returns The potential name.
    */
    static std::string getName()
        {
            return pairEvaluator::getName() + "_" + directionalEnvelope::getName();
        }

    std::string getShapeSpec() const
        {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif


protected:
    Scalar3 dr;
    Scalar rsq;
    Scalar rcutsq;
    pairEvaluator pairEval;           //!< An isotropic pair evaluator
    directionalEnvelope envelEval;    //!< A directional envelope evaluator

};
    
