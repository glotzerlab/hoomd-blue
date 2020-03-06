import glob
import sys

#if argv[1] = "**/*.h" gets list of all h files in current and branching directories
#if argv[1] = "**/*.cc" gets list of all cc files in current and branching directories
#if argv[1] = "**/*.cu" gets list of all cu files in current and branching directories
#if argv[1] = "**/*.cuh" gets list of all cuh files in current and branching directories
dir_names = glob.glob(sys.argv[1], recursive = 'True')


#dictionary of replacements for cuda and hip functions, typenames, and macros
replacements = {'cudaError_t':'gpu::error_t',
                'hipError_t':'gpu::error_t',
                'ENABLE_CUDA':'ENABLE_GPU',
                'ENABLE_HIP':'ENABLE_GPU',
                '__NVCC__':'__GPUCC__',
                '__HIPCC__':'__GPUCC__',
                '__HIP_PLATFORM_NVCC__':'__GPU_PLATFORM_NVIDIA__',
                '__HIP_PLATFORM_HCC__':'__GPU_PLATFORM_AMD__',
                '__HIP_DEVICE_COMPILE__':'__GPU_DEVICE_COMPILE__',
                'hipGetDeviceCount':'gpu::getDeviceCount',
                'hipDeviceProp_t':'gpu::deviceProp_t',
                '<hip/hip_runtime.h>':'<GPURuntime.h>',
                'cuda_runtime.h':'GPURuntime.h',
                'cudaMallocManaged':'gpu::mallocManaged',
                'cudaMemAdvise':'gpu::memAdvise',
                'hipGetErrorString':'gpu::getErrorString',
                'hipDeviceSynchronize':'gpu::deviceSynchronize',
                'hipFree':'gpu::free',
                'hipHostFree':'gpu::hostFree',
                'cudaDeviceSynchronize':'gpu::deviceSynchronize',
                'hipMemcpyAsync':'gpu::memcpyAsync',
                'hipEvent_t':'gpu::event_t',
                'hipGetLastError':'getLastError',
                'CUDADeleter':'GPUDeleter',
                'hip_error_checking':'gpu_error_checking',
                'cudaProfileStart':'gpu::profileStart',
                'cudaProfileStop':'gpu::profileStop',
                'cudaMemAdvise':'gpu::memAdvise',
                'cudaMemAdviseSetReadMostly':'gpu::memAdviseSetReadMostly',
                'cudaMemAdviseSetPreferredLocation':'gpu::memAdviseSetPreferredLocation',
                'cudaCpuDeviceId':'gpu::cpuDeviceId',
                'cudaMemPrefetchAsync':'gpu::memPrefetchAsync',
                'cudaDeviceProp':'gpu::deviceProp_t',
                'cudaSuccess':'gpu::success',
                'cudaMemcpy':'gpu::memcpy',
                'cudaMemset':'gpu::memset',
                'cudaMemsetAsync':'gpu::memsetAsync',
                'cudaPeekAtLastError':'gpu::peekAtLastError',
                'cudaEvent':'gpu::event_t',
                'cudaStreamDefault':'gpu::streamDefault',
                'cudaStreamQuery':'gpu::streamQuery',
                'cudaGetLastError':'gpu::getLastError',
                'cudaSharedMemConfig':'gpu::sharedMemConfig',
                'cudaDeviceSetSharedMemConfig':'gpu::deviceSetSharedMemConfig',
                'cudaEventCreate':'gpu::eventCreate',
                'cudaEventDestroy':'gpu::eventDestroy',
                'cudaEventRecord':'gpu::eventRecord',
                'cudaEventElapsedTime':'gpu::eventElapsedTime',
                'cudaStream_t':'gpu::stream_t',
                'cudaStreamCreate':'gpu::streamCreate',
                'cudaStreamCreateWithFlags':'gpu::streamCreateWithFlags',
                'cudaStreamDestroy':'gpu::streamDestroy',
                'cudaStreamSynchronize':'gpu::streamSynchronize',
                'cudaStreamWaitEvent':'gpu::streamWaitEvent',
                'cudaFree':'gpu::free',
                'cudaEventSynchronize':'gpu::eventSynchronize',
                'cudaGetDeviceProperties':'gpu::getDeviceProperties',
                'cudaMalloc':'gpu::malloc',
                'cudaThreadSynchronize':'gpu::threadSynchronize',
                'cudaSharedMemBankSizeEightByte':'gpu::sharedMemBankSizeEightByte',
                'cudaFuncSetSharedMemConfig':'gpu::funcSetSharedMemConfig',
                'cudaGetErrorString':'gpu::getErrorString',
                'cudaGetDevice':'getDevice',
                'cudaGetDeviceCount':'gpu::getDeviceCount',
                'cudaGetDeviceProperties':'gpu::getDeviceProperties',
                'cudaSetDevice':'gpu::setDevice',
                'cudaSetDeviceFlags':'gpu::setDeviceFlags',
                'cudaDeviceMapHost':'gpu::deviceMapHost',
                'cudaMemAdviseSetAccessedBy':'gpu::memAdviseSetAccessedBy',
                'cudaSetValidDevices':'gpu::setValidDevices',
                'cudaFuncGetAttributes':'gpu::funcGetAttributes',
                'cudaFuncAttributes':'gpu::funcAttributes',
                'cudaMemAttachGlobal':'gpu::memAttachGlobal',
                'cudaMemAttachHost':'gpu::memAttachHost',
                'hipSuccess':'gpu::success',
                'hipGetErrorString':'gpu::getErrorString',
                'hipMallocManaged':'gpu::mallocManaged',
                'hipHostUnregister':'gpu::hostUnregister',
                'hipHostGetDevicePointer':'gpu::hostGetDevicePointer',
                'hipMalloc':'gpu::malloc',
                'hipMemset':'gpu::memset',
                'hipMemsetAsync':'gpu::memsetAsync',
                'hipMemcpyDeviceToHost':'gpu::memcpyDeviceToHost',
                'hipMemcpyAsync':'gpu::memcpyAsync',
                'hipMemcpyHostToDevice':'gpu::memcpyHostToDevice',
                'hipMemcpy':'gpu::memcpy',
                'hipMemAdvise':'gpu::memAdvise',
                'hipHostRegister':'gpu::hostRegister',
                'hipProfilerStart':'gpu::profilerStart',
                'hipProfilerStop':'gpu::profilerStop',
                'hipEventCreate':'gpu::eventCreate',
                'hipEventDestroy':'gpu::eventDestroy',
                'hipEventRecord':'gpu::eventRecord',
                'hipEventElapsedTime':'gpu::eventElapsedTime',
                'hipEventSynchronize':'gpu::eventSynchronize',
                'hipEventCreateWithFlags':'gpu::eventCreateWithFlags',
                'hipEventDisableTiming':'gpu::EventDisableTiming',
                'hipSetDevice':'gpu::setDevice',
                'hipSetDeviceFlags':'gpu::setDeviceFlags',
                'hipDeviceMapHost':'gpu::deviceMapHost',
                'hipDeviceSetLimit':'gpu::deviceSetLimit',
                'hipGetDevice':'gpu::getDevice',
                'hipGetDeviceProperties':'gpu::getDeviceProperties',
                'hipStreamWaitEvent':'gpu::streamWaitEvent',
                'HIP_VERSION_MAJOR':'GPU_VERSION_MAJOR',
                'HIP_VERSION_MINOR':'GPU_VERSION_MINOR',
                'hipStream_t':'gpu::stream_t',
                'hipStreamDestroy':'gpu::streamDestroy',
                'hipStreamCreate':'gpu::streamCreate',
                'hipStreamCreateWithFlags':'gpu::streamCreateWithFlags',
                'hipStreamDefault':'gpu::streamDefault',
                'hipStreamSynchronize':'gpu::streamSynchronize',
                'hipMemcpyDefault':'gpu::memcpyDefault',
                'hipHostMalloc':'gpu::hostMalloc',
                'hipHostMallocMapped':'gpu::hostMallocMapped',
                'hipHostMallocDefault':'gpu::hostMallocDefault',
                '"hip/hip_runtime.h"':'<GPURuntime.h>',
                'hipFuncGetAttributes':'gpu::funcGetAttributes',
                'hipFuncAttributes':'gpu::funcAttributes',
                'hipMemAttachGlobal':'gpu::memAttachGlobal',
                'hipMemAttachHost':'gpu::memAttachHost'}

#variables for formatting kernal launchers and their args
launch_kernel_indent = ""
indenting_kernel_launcher_args = False

#returns indent size of line
def getIndent(line):
   indent_length = 0
   for character in line:
       if not character == " ":
         break;
       indent_length += 1
   return indent_length*" "

#returns name of kernel being launched
def getKernelName(line, indent):
   if "hipLaunchKernelGGL" in line:
       idx_start = line.find("(") + 1
       idx_end = line.find(",")
       split_line = line.split(",", 5)
       name = line[idx_start:idx_end]
       if "HIP_KERNEL_NAME(" in name:
           name = name.replace("HIP_KERNEL_NAME(", "")
           name = name.replace(")", "")
       #first_arg = split_line[5]
       return name + ", " # + first_arg
   elif "<<<" in line:
       idx_end = line.find("<<<")
       name = line[len(indent):idx_end]
       idx_arg_start = line.find("(")
       first_arg = line[idx_arg_start + 1:len(line)]
       return name + ", " + first_arg

def getKernelParams(line):
   if "hipLaunchKernelGGL" in line:
       split_line = line.split()
       param_one = split_line[1]
       param_two = split_line[2]
       param_two = param_two.replace(",", "")
       return param_one + param_two
   if "<<<" in line:
       idx_start = line.find("<<<")
       idx_end = line.find(">>>")
       return line[idx_start + 3:idx_end]

#for loop goes through all files and replace cuda/hip functions, typenames, and MACROS with their gpu namespace defintions... also replaces cuda/hip kernal launchers
for filename in dir_names:
   all_lines = []
   if not "GPURuntime.h" in filename and not "replace.py" in filename and not "search.py" in filename:
       file_read = open(filename, "r")
       for line in file_read:
           updated_line = line
           for definition in replacements:
               if definition in line:
                   updated_line = updated_line.replace(definition, replacements[definition])

           if indenting_kernel_launcher_args == True:
               removal_indent = getIndent(updated_line)
               arg = updated_line[len(removal_indent):len(updated_line) - 1]
               updated_line = launch_kernel_indent + 8*" " + arg + "\n"
               if ");" in line:
                   indenting_kernel_launcher_args = False

           if "<<<" in line or "hipLaunchKernelGGL" in line:
               launch_kernel_indent = getIndent(updated_line)
               kernel_name_and_arg = getKernelName(updated_line, launch_kernel_indent)
               kernel_params = getKernelParams(updated_line)
               launch_kernel_line = launch_kernel_indent + "KernelLauncher launcher(" + kernel_params + ");"
               updated_line = launch_kernel_indent + "launcher(" + kernel_name_and_arg
               all_lines.append(launch_kernel_line + "\n")
               if not ");" in line:
                   indenting_kernel_launcher_args = True

           all_lines.append(updated_line)
       file_read.close()
       file_write = open(filename, "w+")
       file_write.writelines(all_lines)
       file_write.close()

