macro(fix_cudart_rpath target)
if (ENABLE_CUDA AND APPLE)
add_custom_command(TARGET $<TARGET_FILE:${target}> POST_BUILD
                          COMMAND install_name_tool ARGS -change @rpath/libcudart.dylib ${CUDA_CUDART_LIBRARY} ${_target_exe})
add_custom_command(TARGET $<TARGET_FILE:${target}> POST_BUILD
                          COMMAND install_name_tool ARGS -change @rpath/libcufft.dylib ${CUDA_cufft_LIBRARY} ${_target_exe})
endif (ENABLE_CUDA AND APPLE)
endmacro(fix_cudart_rpath)
