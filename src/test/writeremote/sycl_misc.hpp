#include <sycl/sycl.hpp>
#include <iostream>

template <int ndev, int nsub>
sycl::device getSubDevice() {
  static auto devs = sycl::device::get_devices(sycl::info::device_type::gpu);
  auto dev = devs[ndev];
  try {
    static auto subs = dev.template create_sub_devices<
      sycl::info::partition_property::partition_by_affinity_domain>(
          sycl::info::partition_affinity_domain::numa);

    return subs[nsub];
  } catch (sycl::exception &e) {
    std::cout<<e.what()<<std::endl;
    return dev;
  };
}

template <int ndev, int nsub>
sycl::queue getQueue() {
  static sycl::queue q(
      getSubDevice<ndev, nsub>(),
      sycl::property_list {
        sycl::property::queue::enable_profiling(),
        sycl::property::queue::in_order()
      });
  return q;
}

#define queue_case(x) \
case x: \
  if (nsub == 0) \
    return getQueue<x, 0>(); \
  else \
    return getQueue<x, 1>();

sycl::queue currentQueue(int ndev, int nsub) {
  switch(ndev) {
    queue_case(0);
    queue_case(1);
    queue_case(2);
    queue_case(3);
    queue_case(4);
    queue_case(5);
    queue_case(6);
    queue_case(7);
  }
  throw std::exception();
}

#define subdev_case(x) \
case x: \
  if (nsub == 0) \
    return getSubDevice<x, 0>(); \
  else \
    return getSubDevice<x, 1>();

sycl::device currentSubDevice(int ndev, int nsub) {
  switch(ndev) {
    subdev_case(0);
    subdev_case(1);
    subdev_case(2);
    subdev_case(3);
    subdev_case(4);
    subdev_case(5);
    subdev_case(6);
    subdev_case(7);
  }
  throw std::exception();
}

static uint32_t g_dev_num = 1;
static uint32_t g_part_num = 0;

sycl::device currentSubDevice() {
  return currentSubDevice(g_dev_num, g_part_num);
}

sycl::queue currentQueue() {
  return currentQueue(g_dev_num, g_part_num);
}
