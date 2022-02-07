set (_header_files export_filters.h
                   ParticleFilterAll.h
                   ParticleFilterCustom.h
                   ParticleFilter.h
                   ParticleFilterIntersection.h
                   ParticleFilterNull.h
                   ParticleFilterRigid.h
                   ParticleFilterSetDifference.h
                   ParticleFilterTags.h
                   ParticleFilterType.h
                   ParticleFilterUnion.h
           )

install(FILES ${_header_files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd/filter
       )

set(files __init__.py
          filter_.py
          all_.py
          null.py
          rigid.py
          set_.py
          tags.py
          type_.py
          custom.py
          )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/filter
       )

copy_files_to_build("${files}" "filter" "*.py")
