#ifndef __RANDOM_TRIGGER_H__
#define __RANDOM_TRIGGER_H__

#include <memory>
#include <vector>
#include <algorithm>
#include <iterator>
#include <random>
#include <utility>

#include "hoomd/Updater.h"

//! Maintains a list of Updaters with associated weights that are randomly selected for execution
class RandomTrigger
    {
    public:
        RandomTrigger(unsigned int seed)
            : m_seed(seed)
            { }

        bool isEligibleForExecution(unsigned int timestep, const Updater& updater) const
            {
            std::vector<std::shared_ptr<Updater> > cur_members;
            std::vector<double> weights;
            for (auto p: m_moves)
                {
                auto s = p.first.lock();
                if (s)
                    {
                    cur_members.push_back(s);
                    weights.push_back(p.second);
                    }
                }
            // choose a random member of the set by its weight
            if (!cur_members.size()) return false;
            std::seed_seq seed({m_seed, timestep});;
            std::mt19937 rng(seed);


            double total(0.0);
            for (auto w: weights) total += w;
            double r = std::uniform_real_distribution<>(0.0, total)(rng);

            double cur_total(0.0);
            unsigned int k;
            for (k = 0; k < cur_members.size(); ++k)
                {
                cur_total += weights[k];
                if (r <= cur_total)
                    break;
                }

            return cur_members[k].get() == &updater;
            }

        void addToSet(std::shared_ptr<Updater> updater, double weight)
            {
            for (auto p: m_moves)
                {
                auto s = p.first.lock();
                if (s.get() == updater.get())
                    return; // ignore if it is already in the set
                }
            m_moves.push_back(std::make_pair(std::weak_ptr<Updater>(updater),weight));
            }

        void removeFromSet(std::shared_ptr<Updater> updater)
            {
            for (auto it = m_moves.begin(); it != m_moves.end(); ++it)
                {
                auto s = it->first.lock();
                if (s.get() == updater.get())
                    {
                    m_moves.erase(it);
                    break;
                    }
                }
            }

    private:
        unsigned int m_seed;                                                //!< RNG seed
        std::vector<std::pair<std::weak_ptr<Updater>, double> > m_moves;    //!< The list of possible moves and their associated weights
    };

//! Export the RandomTrigger class to python
inline void export_RandomTrigger(pybind11::module& m)
    {
    pybind11::class_< RandomTrigger, std::shared_ptr< RandomTrigger > >(m, "RandomTrigger")
          .def( pybind11::init< unsigned int>())
          .def("addToSet", &RandomTrigger::addToSet)
          .def("removeFromSet", &RandomTrigger::removeFromSet)
          .def("addToSet", &RandomTrigger::addToSet, pybind11::arg("updater"), pybind11::arg("weight") = 1.0)
          ;
    }

#endif // __RANDOM_TRIGGER_H__
