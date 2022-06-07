#include "env.h"
#include <sstream>
#include <iostream>

/*
 * All available launch options for ONECCL_BINDINGS_FOR_PYTORCH
 * ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE:           Default = 0, Set verbose level in ONECCL_BINDINGS_FOR_PYTORCH
 * ONECCL_BINDINGS_FOR_PYTORCH_ENV_WAIT_GDB:          Default = 0, Set 1 to force the oneccl_bindings_for_pytorch wait for GDB attaching
 */

#define ONECCL_BINDINGS_FOR_PYTORCH_ENV_TYPE_DEF(var)                                     \
    int var = [&]() -> int {                                            \
      auto env = std::getenv("ONECCL_BINDINGS_FOR_PYTORCH_" #var);                        \
      int _##var = 0;                                                   \
      try {                                                             \
        _##var = std::stoi(env, 0, 10);                                 \
      } catch (...) { /* Do Nothing */ }                                \
      return _##var;                                                    \
    } ()

int oneccl_bindings_for_pytorch_env(int env_type) {

  static struct {
    ONECCL_BINDINGS_FOR_PYTORCH_ENV_TYPE_DEF(ENV_VERBOSE);
    ONECCL_BINDINGS_FOR_PYTORCH_ENV_TYPE_DEF(ENV_WAIT_GDB);
  } env;

  switch (env_type) {
    case ENV_VERBOSE:
      return env.ENV_VERBOSE;
    case ENV_WAIT_GDB:
      return env.ENV_WAIT_GDB;
    default:
      return 0;
  }
}
