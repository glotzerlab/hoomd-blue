// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_NORMALIZER_H__
#define __PAIR_EVALUATOR_NORMALIZER_H__

#include "hoomd/HOOMDMath.h"

template<class evaluator> class Normalized : public evaluator
    {
    Normalized(typename evaluator::param_type param) : evaluator(param) { }
    Scalar m_normValue = 1.;

    void setNormalizationValue(Scalar value)
        {
        m_normValue = value;
        }

    bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
        bool evaluated = evaluator::evalForceAndEnergy(force_divr, pair_eng, energy_shift);
        force_divr *= m_normValue;
        pair_eng *= m_normValue;
        return evaluated;
        }
    };

#endif // __PAIR_EVALUATOR_NORMALIZER_H__
