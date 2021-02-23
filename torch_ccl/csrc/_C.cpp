#include <init.h>

PYBIND11_MODULE(_C, m) {
  torch_ccl_python_init(m);
}