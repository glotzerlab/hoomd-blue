
// inclusion guard
#ifndef _DEM_GSD_SHAPE_SPEC_
#define _DEM_GSD_SHAPE_SPEC_

template <class Vector>
class DEMGSDShapeSpec
    {
    DEMGSDShapeSpec(const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) :  m_exec_conf(exec_conf), m_mpi(mpi) {}
    const std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
    bool m_mpi;

    //! Method that is called whenever the GSD file is written if connected to a GSD file.
    int slotWriteGSDShapeSpec(gsd_handle&, std::string name) const;

    //! Method that is called to connect to the gsd write state signal
    void connectGSDShapeSpec(std::shared_ptr<GSDDumpWriter> writer, std::string name);

    std::string getShapeSpec()


    }

template <class Vector>
void DEMGSDShapeSpec<Vector>::connectGSDShapeSpec(std::shared_ptr<GSDDumpWriter> writer,
                                          std::string name)
    {
    typedef hoomd::detail::SharedSignalSlot<int(gsd_handle&)> SlotType;
    auto func = std::bind(&DEMGSDShapeSpec<Vector>::slotWriteGSDShapeSpec, this, std::placeholders::_1, name);
    std::shared_ptr<hoomd::detail::SignalSlot> pslot( new SlotType(writer->getWriteSignal(), func));
    addSlot(pslot);
    }

template <class Vector>
int DEMGSDShapeSpec<Vector>::slotWriteGSDShapeSpec( gsd_handle& handle, std::string name ) const
    {
    // m_exec_conf->msg->notice(10) << "DEMGSDShapeSpec writing to GSD File to name: "<< name << std::endl;

    // create schema helpers
    #ifdef ENABLE_MPI
    bool mpi=(bool)m_pdata->getDomainDecomposition();
    #else
    bool mpi=false;
    #endif

    gsd_shape_spec<Shape> schema(m_exec_conf, mpi);
    int retval = schema.write(handle, name, this->m_params);
    return retval;
    }

template <class Vector>
std::string DEMGSDShapeSpec<Vector>::getShapeType(std::vector<Vector<Real> > &verts, unsigned int dim)
    {
    std::string shapetype;
    std::ostringstream shapedef;
    unsigned int nverts = verts.size();
    if (dim == 2)
        {
        if (nverts == 1)
            shapetype = "'Disk'";
        else
            shapetype = "'Polygon'";
        }
    else
        {
        if (nverts == 1)
            shapetype = "'Sphere'";
        else
            shapetype = "'ConvexPolyhedron'";
        }

    if (shapetype == "Disk" || shapetype == "Sphere")
        {
        shapedef =  "{'type': " << shapetype << ", 'diameter': " << 2*radius << "}";
        }
    else
        {
        shapedef =  "{'type': " << shapetype << ", 'rounding_radius': " << radius <<
                    ", 'vertices': "  << parseVertices<Vector>(verts, dim) << "}";
        }
    return shapedef.str();
    }

template <class Vector>
inline std::string DEMGSDShapeSpec<Vector>::parseVertices(std::vector<Vector<Real> > &verts, unsigned int dim)
    {
    std::ostringstream vertstr;
    unsigned int nverts = verts.size();
    for (unsigned int i = 0; i < nverts-1; i++)
        {
        if (dim == 2)
            vertstr << "[ " << verts[i].x << ", " << verts[i].y << "], ";
        else
            vertstr << "[ " << verts[i].x << ", " << verts[i].y << ", " << verts[i].z << "], ";
        }
    if (dim == 2)
        vertstr << "[ " << verts[nverts-1].x << ", " << verts[nverts-1].y << "]" << " ]" << "}";
    else
        vertstr << "[ " << verts[nverts-1].x << ", " << verts[nverts-1].y << ", " << verts[i].z << "]" << " ]" << "}";
    return vertstr.str();
    }

template <class Vector>
std::vector<std::string> DEMGSDShapeSpec<Vector>::getShapeSpec()
    {
    unsigned int ntypes = m_shapes.size();
    std::vector<std::string> shapespec(ntypes);
    for (unsigned int i = 0; i < ntypes; i++)
        shapespec[i] = getShapeType<Vector>(verts, dimensions);
    return shapespec;
    }

#endif
