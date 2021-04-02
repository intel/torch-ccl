#pragma once

enum TORCH_CCL_ENV {
  ENV_VERBOSE = 0,
  ENV_WAIT_GDB
};

int torch_ccl_env(int env);

static inline int torch_ccl_verbose() {
  return torch_ccl_env(ENV_VERBOSE);
}

static inline int torch_ccl_wait_gdb() {
  return torch_ccl_env(ENV_WAIT_GDB);
}