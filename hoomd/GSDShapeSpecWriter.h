// inclusion guard
#ifndef __GSD_SHAPE_SPEC_WRITER
#define __GSD_SHAPE_SPEC_WRITER

#include "extern/gsd.h"
#include "GSDDumpWriter.h"
#include <sstream>
#include <iostream>

class GSDShapeSpecWriter
    {
    public:

        GSDShapeSpecWriter(const std::shared_ptr<const ExecutionConfiguration> exec_conf) : m_exec_conf(exec_conf) {};

        int write(gsd_handle& handle, const std::vector<std::string> &type_shape_mapping)
            {
            if(!m_exec_conf->isRoot())
                return 0;
            std::string name = "particles/type_shapes";
            int max_len = 0;
            for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
                max_len = std::max(max_len, (int)type_shape_mapping[i].size());
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

    private:

        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;

    };

#endif
