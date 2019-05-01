// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file Profiler.cc
    \brief Defines the Profiler class
*/

#include "Profiler.h"

#include <iomanip>
#include <sstream>


using namespace std;
namespace py = pybind11;


////////////////////////////////////////////////////
// ProfileDataElem members
int64_t ProfileDataElem::getChildElapsedTime() const
    {
    // start counting the elapsed time from our time
    int64_t total = 0;

    // for each of the children
    map<string, ProfileDataElem>::const_iterator i;
    for (i = m_children.begin(); i != m_children.end(); ++i)
        {
        // add their time
        total += (*i).second.m_elapsed_time;
        }

    // return the total
    return total;
    }
int64_t ProfileDataElem::getTotalFlopCount() const
    {
    // start counting the elapsed time from our time
    int64_t total = m_flop_count;

    // for each of the children
    map<string, ProfileDataElem>::const_iterator i;
    for (i = m_children.begin(); i != m_children.end(); ++i)
        {
        // add their time
        total += (*i).second.getTotalFlopCount();
        }

    // return the total
    return total;
    }
int64_t ProfileDataElem::getTotalMemByteCount() const
    {
    // start counting the elapsed time from our time
    int64_t total = m_mem_byte_count;

    // for each of the children
    map<string, ProfileDataElem>::const_iterator i;
    for (i = m_children.begin(); i != m_children.end(); ++i)
        {
        // add their time
        total += (*i).second.getTotalMemByteCount();
        }

    // return the total
    return total;
    }

/*! Recursive output routine to write results from this profile node and all sub nodes printed in
    a tree.
    \param o stream to write output to
    \param name Name of the node
    \param tab_level Current number of tabs in the tree
    \param total_time Total number of nanoseconds taken by this node
    \param name_width Maximum name width for all siblings of this node (used to align output columns)
 */
void ProfileDataElem::output(std::ostream &o, const std::string& name, int tab_level, int64_t total_time, int name_width) const
    {
    // create a tab string to output for the current tab level
    string tabs = "";
    for (int i = 0; i < tab_level; i++)
        tabs += "        ";

    o << tabs;
    // start with an overview
    // initial tests determined that having a parent node calculate the avg gflops of its
    // children is annoying, so default to 0 flops&bytes unless we are a leaf
    double sec = double(m_elapsed_time)/1e9;
    double perc = double(m_elapsed_time)/double(total_time) * 100.0;
    double flops = 0.0;
    double bytes = 0.0;
    if (m_children.size() == 0)
        {
        flops = double(getTotalFlopCount())/sec;
        bytes = double(getTotalMemByteCount())/sec;
        }

    output_line(o, name, sec, perc, flops, bytes, name_width);

    // start by determining the name width
    map<string, ProfileDataElem>::const_iterator i;

    // it has at least to be four letters wide ("Self")
    int child_max_width = 4;
    for (i = m_children.begin(); i != m_children.end(); ++i)
        {
        int child_width = (int)(*i).first.size();
        if (child_width > child_max_width)
            child_max_width = child_width;
        }

    // output each of the children
    for (i = m_children.begin(); i != m_children.end(); ++i)
        {
        (*i).second.output(o, (*i).first, tab_level+1, total_time, child_max_width);
        }

    // output an "Self" item to account for time actually spent in this data elem
    if (m_children.size() > 0)
        {
        double sec = double(m_elapsed_time - getChildElapsedTime())/1e9;
        double perc = double(m_elapsed_time - getChildElapsedTime())/double(total_time) * 100.0;
        double flops = double(m_flop_count)/sec;
        double bytes = double(m_mem_byte_count)/sec;

        // don't print Self unless perc is significant
        if (perc >= 0.1)
            {
            o << tabs << "        ";
            output_line(o, "Self", sec, perc, flops, bytes, child_max_width);
            }
        }
    }

void ProfileDataElem::output_line(std::ostream &o,
                                  const std::string &name,
                                  double sec,
                                  double perc,
                                  double flops,
                                  double bytes,
                                  unsigned int name_width) const
    {
    o << setiosflags(ios::fixed);

    o << name << ": ";
    assert(name_width >= name.size());
    for (unsigned int i = 0; i < name_width - name.size(); i++)
        o << " ";

    o << setw(7) << setprecision(4) << sec << "s";
    o << " | " << setprecision(3) << setw(6) << perc << "% ";

    //If sec is zero, the values to be printed are garbage.  Thus, we skip it all together.
    if (sec == 0)
        {
        o << "n/a" << endl;
        return;
        }

    o << setprecision(5);
    // output flops with intelligent units
    if (flops > 0)
        {
        o << setw(6);
        if (flops < 1e6)
            o << flops << "  FLOP/s ";
        else if (flops < 1e9)
            o << flops/1e6 << " MFLOP/s ";
        else
            o << flops/1e9 << " GFLOP/s ";
        }

    //output bytes/s with intelligent units
    if (bytes > 0)
        {
        o << setw(6);
        if (bytes < 1e6)
            o << bytes << "  B/s ";
        else if (bytes < 1e9)
            o << bytes/1e6 << " MiB/s ";
        else
            o << bytes/1e9 << " GiB/s ";
        }

    o << endl;
    }

////////////////////////////////////////////////////////////////////
// Profiler

Profiler::Profiler(const std::string& name) : m_name(name)
    {
    // push the root onto the top of the stack so that it is the default
    m_stack.push(&m_root);

    // record the start of this profile
    m_root.m_start_time = m_clk.getTime();

    #ifdef SCOREP_USER_ENABLE
    SCOREP_USER_REGION_BEGIN(m_root.m_scorep_region, name.c_str(),SCOREP_USER_REGION_TYPE_COMMON )
    #endif
    }

void Profiler::output(std::ostream &o)
    {
    // perform a sanity check, but don't bail out
    if (m_stack.top() != &m_root)
        {
        o << "***Warning! Outputting a profile with incomplete samples" << endl;
        }

    #ifdef SCOREP_USER_ENABLE
    SCOREP_USER_REGION_END( m_root.m_scorep_region )
    #endif

    // outputting a profile implicitly calls for a time sample
    m_root.m_elapsed_time = m_clk.getTime() - m_root.m_start_time;

    // startup the recursive output process
    m_root.output(o, m_name, 0, m_root.m_elapsed_time, (int)m_name.size());
    }

/*! \param o Stream to output to
    \param prof Profiler to print
*/
std::ostream& operator<<(ostream &o, Profiler& prof)
    {
    prof.output(o);

    return o;
    }

//! Helper function to get the formatted output of a Profiler in python
/*! Outputs the profiler timings to a string
    \param prof Profiler to generate output from
*/
string print_profiler(Profiler *prof)
    {
    assert(prof);
    ostringstream s;
    s << *prof;
    return s.str();
    }

void export_Profiler(py::module& m)
    {
    py::class_<Profiler>(m,"Profiler")
    .def(py::init<const std::string&>())
    .def("__str__", &print_profiler)
    ;
    }
