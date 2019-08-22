#include "extern/gsd.h"
#include "GSDDumpWriter.h"
#include "GSDReader.h"
#include "HOOMDMPI.h"

#include <string>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#ifndef _GSD_SHAPE_SPEC_H_
#define _GSD_SHAPE_SPEC_H_

template<class T>
using param_array = typename GlobalArray<T>;

struct gsd_schema_hpmc_base
    {
    gsd_schema_hpmc_base(const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) : m_exec_conf(exec_conf), m_mpi(mpi) {}
    const std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
    bool m_mpi;
    };

template < class Evaluator >
struct gsd_shape_spec
    {
    gsd_shape_spec(const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) : gsd_schema_hpmc_base(exec_conf, mpi) {}

    int write(gsd_handle& handle, const std::string& name, const param_array<typename Evaluator::param_type > &params)
        {
        if(!m_exec_conf->isRoot())
            return 0;

        std::vector< std::string > type_shape_mapping(params.getNumElements());
        Scalar3 dr = make_scalar3(0,0,0);
        Scalar4 quat = make_scalar4(0,0,0,0);
        Scalar rcutsq = 0;
        int max_len = 0;
        for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
            {
            Evaluator evaluator(dr, quat, quat, rcutsq, params[i]);
            type_shape_mapping[i] = evaluator.getShapeSpec();
            max_len = std::max(max_len, (int)type_shape_mapping[i].size());
            }
        max_len += 1;  // for null
        m_exec_conf->msg->notice(10) << "dump.gsd: writing " << name << std::endl;
        std::vector<char> types(max_len * type_shape_mapping.size());
        for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
            strncpy(&types[max_len*i], type_shape_mapping[i].c_str(), max_len);
        int retval = gsd_write_chunk(&handle, name.c_str(), GSD_TYPE_UINT8, type_shape_mapping.size(), max_len, 0, (void *)&types[0]);
        if (retval == -1)
            {
            m_exec_conf->msg->error() << "dump.gsd: " << strerror(errno) << std::endl;
            throw std::runtime_error("Error writing GSD file");
            }
        else if (retval != 0)
            {
            m_exec_conf->msg->error() << "dump.gsd: " << "Unknown error " << retval << std::endl;
            throw std::runtime_error("Error writing GSD file");
            }
        return retval;
        }
    };
#endif
