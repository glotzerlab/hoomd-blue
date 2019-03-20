// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __DUMPITERATORS_H_
#define __DUMPITERATORS_H_

#include <iterator>
#include <map>
#include <stdint.h>

#include "hoomd/VectorMath.h"
#include "hoomd/BondedGroupData.h"

/// This file contains iterators to serialize and deserialize various
/// composite data types found in hoomd for use in the getar reader
/// and writer to and from flat streams of data.

namespace getardump{

    // Iterator to give the keys of a map
    template<typename Key, typename Val>
    class MapValueIterator: public std::iterator<std::input_iterator_tag, Val>
        {
        public:
            typedef std::map<Key, Val> map_t;

            MapValueIterator():
                m_begin()
                {}

            MapValueIterator(typename map_t::iterator begin):
                m_begin(begin)
                {}

            void operator++() {++m_begin;}

            Val &operator*() const
                {
                return m_begin->second;
                }

            Val *operator->() const
                {
                return &(m_begin->second);
                }

            bool operator==(const MapValueIterator<Key, Val> &rhs) const
                {
                return m_begin == rhs.m_begin;
                }

            bool operator!=(const MapValueIterator<Key, Val> &rhs) const
                {return !(rhs == *this);}

        private:
            typename map_t::iterator m_begin;
        };

    // Special iterator classes for serializing ArrayHandle data
    template<typename Real, typename Iter>
    class Scalar4xyzIterator: public std::iterator<std::input_iterator_tag, Real>
        {
        public:
            Scalar4xyzIterator(const Iter &begin):
                m_major(begin), m_minor(0)
                {}

            void operator++()
                {
                if(++m_minor == 3)
                    ++m_major;
                m_minor %= 3;
                }

            float operator*()
                {
                switch(m_minor)
                    {
                    case 0:
                        return (Real) m_major->x;
                    case 1:
                        return (Real) m_major->y;
                    case 2:
                    default:
                        return (Real) m_major->z;
                    }
                }

            bool operator==(const Scalar4xyzIterator &rhs) const
                {
                return m_major == rhs.m_major && m_minor == rhs.m_minor;
                }

            bool operator!=(const Scalar4xyzIterator &rhs) const
                {return !(rhs == *this);}

        private:
            Iter m_major;
            unsigned int m_minor;
        };

    template<typename Real, typename Iter>
    class Scalar3xyzIterator: public std::iterator<std::input_iterator_tag, Real>
        {
        public:
            Scalar3xyzIterator(const Iter &begin):
                m_major(begin), m_minor(0)
                {}

            void operator++()
                {
                if(++m_minor == 3)
                    ++m_major;
                m_minor %= 3;
                }

            float operator*()
                {
                switch(m_minor)
                    {
                    case 0:
                        return (Real) m_major->x;
                    case 1:
                        return (Real) m_major->y;
                    case 2:
                    default:
                        return (Real) m_major->z;
                    }
                }

            bool operator==(const Scalar3xyzIterator &rhs) const
                {
                return m_major == rhs.m_major && m_minor == rhs.m_minor;
                }

            bool operator!=(const Scalar3xyzIterator &rhs) const
                {return !(rhs == *this);}

        private:
            Iter m_major;
            unsigned int m_minor;
        };

    template<typename Iter>
    class Int3xyzIterator: public std::iterator<std::input_iterator_tag, int32_t>
        {
        public:
            Int3xyzIterator(const Iter &begin):
                m_major(begin), m_minor(0)
                {}

            void operator++()
                {
                if(++m_minor == 3)
                    ++m_major;
                m_minor %= 3;
                }

            int operator*()
                {
                switch(m_minor)
                    {
                    case 0:
                        return (int32_t) m_major->x;
                    case 1:
                        return (int32_t) m_major->y;
                    case 2:
                    default:
                        return (int32_t) m_major->z;
                    }
                }

            bool operator==(const Int3xyzIterator &rhs) const
                {
                return m_major == rhs.m_major && m_minor == rhs.m_minor;
                }

            bool operator!=(const Int3xyzIterator &rhs) const
                {return !(rhs == *this);}

        private:
            Iter m_major;
            unsigned int m_minor;
        };

    template<typename Real, typename Iter>
    class Scalar4xyzwIterator: public std::iterator<std::input_iterator_tag, Real>
        {
        public:
            Scalar4xyzwIterator(const Iter &begin):
                m_major(begin), m_minor(0)
                {}

            void operator++()
                {
                if(++m_minor == 4)
                    ++m_major;
                m_minor %= 4;
                }

            float operator*()
                {
                switch(m_minor)
                    {
                    case 0:
                        return (Real) m_major->x;
                    case 1:
                        return (Real) m_major->y;
                    case 2:
                        return (Real) m_major->z;
                    case 3:
                    default:
                        return (Real) m_major->w;
                    }
                }

            bool operator==(const Scalar4xyzwIterator &rhs) const
                {
                return m_major == rhs.m_major && m_minor == rhs.m_minor;
                }

            bool operator!=(const Scalar4xyzwIterator &rhs) const
                {return !(rhs == *this);}

        private:
            Iter m_major;
            unsigned int m_minor;
        };

    template<typename Real, typename Iter>
    class QuatsxyzIterator: public std::iterator<std::input_iterator_tag, Real>
        {
        public:
            QuatsxyzIterator(const Iter &begin):
                m_major(begin), m_minor(0)
                {}

            void operator++()
                {
                if(++m_minor == 4)
                    ++m_major;
                m_minor %= 4;
                }

            float operator*()
                {
                switch(m_minor)
                    {
                    case 0:
                        return (Real) m_major->s;
                    case 1:
                        return (Real) m_major->v.x;
                    case 2:
                        return (Real) m_major->v.y;
                    case 3:
                    default:
                        return (Real) m_major->v.z;
                    }
                }

            bool operator==(const QuatsxyzIterator &rhs) const
                {
                return m_major == rhs.m_major && m_minor == rhs.m_minor;
                }

            bool operator!=(const QuatsxyzIterator &rhs) const
                {return !(rhs == *this);}

        private:
            Iter m_major;
            unsigned int m_minor;
        };

    template<typename Iter>
    class Scalar4wIterator: public std::iterator<std::input_iterator_tag, float>
        {
        public:
            Scalar4wIterator(const Iter &begin):
                m_major(begin)
                {}

            void operator++()
                {
                ++m_major;
                }

            float operator*()
                {
                return m_major->w;
                }

            bool operator==(const Scalar4wIterator &rhs) const
                {
                return m_major == rhs.m_major;
                }

            bool operator!=(const Scalar4wIterator &rhs) const
                {return !(rhs == *this);}

        private:
            Iter m_major;
        };

    template<typename Iter>
    class Scalar4wToIntIterator: public std::iterator<std::input_iterator_tag, uint32_t>
        {
        public:
            Scalar4wToIntIterator(const Iter &begin):
                m_major(begin)
                {}

            void operator++()
                {
                ++m_major;
                }

            unsigned int operator*()
                {
                return (uint32_t) __scalar_as_int(m_major->w);
                }

            bool operator==(const Scalar4wToIntIterator &rhs) const
                {
                return m_major == rhs.m_major;
                }

            bool operator!=(const Scalar4wToIntIterator &rhs) const
                {return !(rhs == *this);}

        private:
            Iter m_major;
        };

    template<typename Real, typename Iter>
    class VirialIterator: public std::iterator<std::input_iterator_tag, Real>
        {
        public:
            VirialIterator(Iter *begin):
                m_majors(begin), m_minor(0)
                {}

            void operator++()
                {
                ++m_minor;
                m_minor %= 6;
                ++m_majors[m_minor];
                }

            float operator*()
                {
                return (Real) *(m_majors[m_minor]);
                }

            bool operator==(const VirialIterator &rhs) const
                {
                bool result(m_minor == rhs.m_minor);
                for(unsigned int i(0); i < 6; ++i)
                    result &= m_majors[i] == rhs.m_majors[i];
                return result;
                }

            bool operator!=(const VirialIterator &rhs) const
                {return !(rhs == *this);}

        private:
            Iter *m_majors;
            unsigned int m_minor;
        };

    template<typename Real>
    class InvInertiaTensorIterator: public std::iterator<std::input_iterator_tag, vec3<Scalar> >
        {
        public:
            InvInertiaTensorIterator(const typename std::vector<Real>::iterator &begin):
                m_ptr(begin)
                {}

            void operator++() {m_ptr += 6;}

            vec3<Scalar> operator*()
                {return vec3<Scalar>(m_ptr[0], m_ptr[3], m_ptr[5]);}

            bool operator==(const InvInertiaTensorIterator &rhs) const
                {return m_ptr == rhs.m_ptr;}

            bool operator!=(const InvInertiaTensorIterator &rhs) const
                {return !(rhs == *this);}

        private:
            typename std::vector<Real>::iterator m_ptr;
        };

    template<typename From, typename To>
    class TypecastIterator: public std::iterator<std::input_iterator_tag, To>
        {
        public:
            TypecastIterator(From *begin):
                m_ptr(begin)
                {}

            void operator++() {++m_ptr;}

            To operator*() {return (To) *m_ptr;}

            bool operator==(const TypecastIterator<From, To> &rhs) const
                {return m_ptr == rhs.m_ptr;}

            bool operator!=(const TypecastIterator<From, To> &rhs) const
                {return !(rhs == *this);}

        private:
            From *m_ptr;
        };

    template<typename Int>
    class int3Iterator: public std::iterator<std::input_iterator_tag, int3>
        {
        public:
            int3Iterator(Int *begin):
                m_ptr(begin)
                {}

            void operator++() {m_ptr += 3;}

            int3 operator*()
                {
                return make_int3(m_ptr[0], m_ptr[1], m_ptr[2]);
                }

            bool operator==(const int3Iterator<Int> &rhs) const
                {return m_ptr == rhs.m_ptr;}

            bool operator!=(const int3Iterator<Int> &rhs) const
                {return !(rhs == *this);}

        private:
            Int *m_ptr;
        };

    template<typename Real>
    class Scalar3Iterator: public std::iterator<std::input_iterator_tag, Scalar3>
        {
        public:
            Scalar3Iterator(Real *begin):
                m_ptr(begin)
                {}

            void operator++() {m_ptr += 3;}

            Scalar3 operator*()
                {
                return make_scalar3(m_ptr[0], m_ptr[1], m_ptr[2]);
                }

            bool operator==(const Scalar3Iterator<Real> &rhs) const
                {return m_ptr == rhs.m_ptr;}

            bool operator!=(const Scalar3Iterator<Real> &rhs) const
                {return !(rhs == *this);}

        private:
            Real *m_ptr;
        };

    template<typename Real>
    class Scalar4Iterator: public std::iterator<std::input_iterator_tag, Scalar4>
        {
        public:
            Scalar4Iterator(Real *begin):
                m_ptr(begin)
                {}

            void operator++() {m_ptr += 4;}

            Scalar4 operator*()
                {
                return make_scalar4(m_ptr[0], m_ptr[1], m_ptr[2], m_ptr[3]);
                }

            bool operator==(const Scalar4Iterator<Real> &rhs) const
                {return m_ptr == rhs.m_ptr;}

            bool operator!=(const Scalar4Iterator<Real> &rhs) const
                {return !(rhs == *this);}

        private:
            Real *m_ptr;
        };

    template<typename Real>
    class Vec3Iterator: public std::iterator<std::input_iterator_tag, vec3<Scalar> >
        {
        public:
            Vec3Iterator(Real *begin):
                m_ptr(begin)
                {}

            void operator++() {m_ptr += 3;}

            vec3<Scalar> operator*()
                {
                return vec3<Scalar>(m_ptr[0], m_ptr[1], m_ptr[2]);
                }

            bool operator==(const Vec3Iterator<Real> &rhs) const
                {return m_ptr == rhs.m_ptr;}

            bool operator!=(const Vec3Iterator<Real> &rhs) const
                {return !(rhs == *this);}

        private:
            Real *m_ptr;
        };

    template<typename Real>
    class QuatIterator: public std::iterator<std::input_iterator_tag, quat<Scalar> >
        {
        public:
            QuatIterator(Real *begin):
                m_ptr(begin)
                {}

            void operator++() {m_ptr += 4;}

            quat<Scalar> operator*()
                {
                return quat<Scalar>(m_ptr[0], vec3<Scalar>(m_ptr[1], m_ptr[2], m_ptr[3]));
                }

            bool operator==(const QuatIterator<Real> &rhs) const
                {return m_ptr == rhs.m_ptr;}

            bool operator!=(const QuatIterator<Real> &rhs) const
                {return !(rhs == *this);}

        private:
            Real *m_ptr;
        };

    template<unsigned int N>
    class GroupTagIterator: public std::iterator<std::input_iterator_tag, unsigned int>
        {
        public:
            GroupTagIterator(const typename std::vector<group_storage<N> >::iterator &begin):
                m_major(begin), m_minor(0)
                {}

            void operator++()
                {
                if(!(++m_minor %= N))
                    ++m_major;
                }

            unsigned int operator*()
                {return m_major->tag[m_minor];}

            bool operator==(const GroupTagIterator<N> &rhs) const
                {return m_major == rhs.m_major && m_minor == rhs.m_minor;}

            bool operator!=(const GroupTagIterator<N> &rhs) const
                {return !(rhs == *this);}

        private:
            typename std::vector<group_storage<N> >::iterator m_major;
            unsigned int m_minor;
        };

    template<unsigned int N>
    class InvGroupTagIterator: public std::iterator<std::input_iterator_tag, group_storage<N> >
        {
        public:
            InvGroupTagIterator(const std::vector<unsigned int>::iterator &begin):
                m_ptr(begin)
                {}

            void operator++()
                {m_ptr += N;}

            group_storage<N> operator*()
                {
                group_storage<N> result;
                for(size_t i(0); i < N; ++i)
                    result.tag[i] = m_ptr[i];
                return result;
                }

            bool operator==(const InvGroupTagIterator<N> &rhs) const
                {return m_ptr == rhs.m_ptr;}

            bool operator!=(const InvGroupTagIterator<N> &rhs) const
                {return !(rhs == *this);}

        private:
            std::vector<unsigned int>::iterator m_ptr;
        };

}

#endif
