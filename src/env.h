#pragma once

enum ONECCL_BINDINGS_FOR_PYTORCH_ENV {
  ENV_VERBOSE = 0,
  ENV_WAIT_GDB
};

int oneccl_bindings_for_pytorch_env(int env);

static inline int oneccl_bindings_for_pytorch_verbose() {
  return oneccl_bindings_for_pytorch_env(ENV_VERBOSE);
}

static inline int oneccl_bindings_for_pytorch_wait_gdb() {
  return oneccl_bindings_for_pytorch_env(ENV_WAIT_GDB);
}