// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#ifndef __GSDSHAPESPECWRITER_H__
#define __GSDSHAPESPECWRITER_H__

#include "extern/gsd.h"
#include "GSDDumpWriter.h"
#include <sstream>
#include <iostream>

class GSDShapeSpecWriter
    {

    public:

        GSDShapeSpecWriter(const std::shared_ptr<const ExecutionConfiguration> exec_conf, std::string name = "particles/type_shapes") :
                           m_exec_conf(exec_conf), m_field_name(name){};

        int write(gsd_handle& handle, const std::vector<std::string> &type_shape_mapping)
            {
            if(!m_exec_conf->isRoot())
                return 0;
            int max_len = getMaxLen(type_shape_mapping);
            m_exec_conf->msg->notice(10) << "dump.gsd: writing " << m_field_name << std::endl;
            std::vector<char> types(max_len * type_shape_mapping.size());
            for (unsigned int i = 0; i < type_shape_mapping.size(); i++)
                strncpy(&types[max_len*i], type_shape_mapping[i].c_str(), max_len);
            int retval = gsd_write_chunk(&handle, m_field_name.c_str(), GSD_TYPE_UINT8, type_shape_mapping.size(), max_len, 0, (void *)&types[0]);
            checkError(retval);
            return retval;
            }

        inline int getMaxLen(const std::vector<std::string> &stringvector)
            {
            int max_len = 0;
            for (unsigned int i = 0; i < stringvector.size(); i++)
                max_len = std::max(max_len, (int)stringvector[i].size());
            return max_len + 1;  // for null
            }

        void checkError(int number)
            {
            if (number == -1)
                {
                this->m_exec_conf->msg->error() << "dump.gsd: " << strerror(errno) << std::endl;
                throw std::runtime_error("Error writing GSD file");
                }
            else if (number != 0)
                {
                this->m_exec_conf->msg->error() << "dump.gsd: " << "Unknown error " << number << std::endl;
                throw std::runtime_error("Error writing GSD file");
                }
            }

      std::string getName()
          {
          return m_field_name;
          }

    private:

        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
        std::string m_field_name;

    };

#endif
