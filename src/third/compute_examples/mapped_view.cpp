#include <iostream>
#include <vector>

#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/reduce.hpp>
#include <boost/compute/container/mapped_view.hpp>

namespace compute = boost::compute;

// this example demonstrates how to use the mapped_view class to map
// an array of numbers to device memory and use the reduce() algorithm
// to calculate the sum.
int main()
{
    // get default device and setup context
    compute::device gpu = compute::system::default_device();
    compute::context context(gpu);
    compute::command_queue queue(context, gpu);
    std::cout << "device: " << gpu.name() << std::endl;

    // create data on host
    int data[] = { 4, 2, 3, 7, 8, 9, 1, 6 };

    // create mapped view on device
    compute::mapped_view<int> view(data, 8, context);

    // use reduce() to calculate sum on the device
    int sum = 0;
    compute::reduce(view.begin(), view.end(), &sum, queue);

    // print the sum on the host
    std::cout << "sum: " << sum << std::endl;

    return 0;
}
