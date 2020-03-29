set (_header_files filter/ParticleFilter.h
                   filter/ParticleFilterAll.h
                   filter/ParticleFilterIntersection.h
                   filter/ParticleFilterSetDifference.h
                   filter/ParticleFilterTags.h
                   filter/ParticleFilterType.h
                   filter/ParticleFilterUnion.h
                   filter/export_filters.h
           )

install(FILES ${_headers_files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd/filter
       )

set(files __init__.py
          filter_.py
          all_.py
          set_.py
          tags.py
          type_.py
          )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/filter
       )

copy_files_to_build("${files}" "filter" "*.py")
