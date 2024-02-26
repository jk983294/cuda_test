#include <iostream>

#include <boost/compute/core.hpp>

namespace compute = boost::compute;

int main()
{
    // get the default device
    compute::device device = compute::system::default_device();

    // print the device's name and platform
    std::cout << "hello from " << device.name();
    std::cout << " (platform: " << device.platform().name() << ")" << std::endl;

    std::cout << "  global memory size: "
              << device.get_info<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE) / 1024 / 1024
              << " MB"
              << std::endl;
    std::cout << "  local memory size: "
              << device.get_info<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE) / 1024
              << " KB"
              << std::endl;
    std::cout << "  constant memory size: "
              << device.get_info<cl_ulong>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE) / 1024
              << " KB"
              << std::endl;

    std::vector<compute::platform> platforms = compute::system::platforms();

    for(size_t i = 0; i < platforms.size(); i++){
        const compute::platform &platform = platforms[i];

        std::cout << "Platform '" << platform.name() << "'" << std::endl;

        std::vector<compute::device> devices = platform.devices();
        for(size_t j = 0; j < devices.size(); j++){
            const compute::device &device = devices[j];

            std::string type;
            if(device.type() & compute::device::gpu)
                type = "GPU Device";
            else if(device.type() & compute::device::cpu)
                type = "CPU Device";
            else if(device.type() & compute::device::accelerator)
                type = "Accelerator Device";
            else
                type = "Unknown Device";

            std::cout << "  " << type << ": " << device.name() << std::endl;
        }
    }

    return 0;
}
//]
