
// inclusion guard
#ifndef _DEM_GSD_SHAPE_SPEC_
#define _DEM_GSD_SHAPE_SPEC_

template <typename Vector, typename Potential>
class DEMGSDShapeSpec
    {
    DEMGSDShapeSpec(const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) :  m_exec_conf(exec_conf), m_mpi(mpi) {}
    const std::shared_ptr<const ExecutionConfiguration> m_exec_conf;
    bool m_mpi;

    //! Method that is called whenever the GSD file is written if connected to a GSD file.
    int slotWriteDEMGSDShapeSpec(gsd_handle&, std::string name) const;

    //! Method that is called to connect to the gsd write state signal
    void connectDEMGSDShapeSpec(std::shared_ptr<GSDDumpWriter> writer, std::string name);

    std::string getShapeType(std::vector<Vector<Real> > &verts, Potential &potential);

    inline std::string parseVertices(std::vector<Vector<Real> > &verts);

    }

template <typename Vector, typename Potential>
void DEMGSDShapeSpec<Vector,Potential>::connectDEMGSDShapeSpec(std::shared_ptr<GSDDumpWriter> writer,
                                          std::string name)
    {
    typedef hoomd::detail::SharedSignalSlot<int(gsd_handle&)> SlotType;
    auto func = std::bind(&DEMGSDShapeSpec<Vector,Potential>::slotWriteDEMGSDShapeSpec, this, std::placeholders::_1, name);
    std::shared_ptr<hoomd::detail::SignalSlot> pslot( new SlotType(writer->getWriteSignal(), func));
    addSlot(pslot);
    }

template <typename Vector, typename Potential>
int DEMGSDShapeSpec<Vector,Potential>::slotWriteDEMGSDShapeSpec( gsd_handle& handle, std::string name ) const
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


template <typename Potential>
std::string DEMGSDShapeSpec<vec2, Potential>::getShapeType(std::vector<vec2<Real> > &verts, Potential &potential)
    {
    std::string shapetype;
    std::ostringstream shapedef;
    unsigned int nverts = verts.size();

    if (nverts == 1)
        {
        shapetype = "'Disk'";
        shapedef =  "{'type': " << shapetype << ", 'diameter': " << 2*potential.getRadius() << "}";
        }
    else
        {
        shapetype = "'Polygon'";
        shapedef =  "{'type': " << shapetype << ", 'rounding_radius': " << potential.getRadius() <<
                    ", 'vertices': "  << parseVertices<vec2,Potential>(potential.getVerts()) << "}";
        }
    return shapedef.str();
    }

template <typename Potential>
std::string DEMGSDShapeSpec<vec3, Potential>::getShapeType(std::vector<vec3<Real> > &verts, Potential &potential)
    {
    std::string shapetype;
    std::ostringstream shapedef;
    unsigned int nverts = verts.size();

    if (nverts == 1)
        {
        shapetype = "'Sphere'";
        shapedef =  "{'type': " << shapetype << ", 'diameter': " << 2*potential.getRadius() << "}";
        }
    else
        {
        shapetype = "'ConvexPolyhedron'";
        shapedef =  "{'type': " << shapetype << ", 'rounding_radius': " << potential.getRadius() <<
                    ", 'vertices': "  << parseVertices<vec3,Potential>(potential.getVerts()) << "}";
        }
    return shapedef.str();
    }

template <typename Potential>
inline std::string DEMGSDShapeSpec<vec2, Potential>::parseVertices(std::vector<vec2<Real> > &verts)
    {
    std::ostringstream vertstr;
    unsigned int nverts = verts.size();
    for (unsigned int i = 0; i < nverts-1; i++)
        {
        vertstr << "[ " << verts[i].x << ", " << verts[i].y << "], ";
        }
    vertstr << "[ " << verts[nverts-1].x << ", " << verts[nverts-1].y << "]" << " ]";
    return vertstr.str();
    }

template <typename Potential>
inline std::string DEMGSDShapeSpec<vec3, Potential>::parseVertices(std::vector<vec3<Real> > &verts)
    {
    std::ostringstream vertstr;
    unsigned int nverts = verts.size();
    for (unsigned int i = 0; i < nverts-1; i++)
        {
        vertstr << "[ " << verts[i].x << ", " << verts[i].y << ", " << verts[i].z << "], ";
        }
    vertstr << "[ " << verts[nverts-1].x << ", " << verts[nverts-1].y << ", " << verts[nverts-1].x  "]" << " ]";
    return vertstr.str();
    }

template <typename Vector, typename Potential>
std::vector<std::string> DEMGSDShapeSpec<Vector,Potential>::getShapeSpec(std::vector<std::vector<Vector<Real> > > shapes)
    {
    unsigned int ntypes = shapes.size();
    std::vector<std::string> shapespec(ntypes);
    for (unsigned int i = 0; i < ntypes; i++)
        shapespec[i] = getShapeType<Vector,Potential>(verts, potential);
    return shapespec;
    }

#endif
