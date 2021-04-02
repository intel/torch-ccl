#include "env.h"
#include <sstream>
#include <iostream>

/*
 * All available launch options for TORCH_CCL
 * TORCH_CCL_ENV_VERBOSE:           Default = 0, Set verbose level in TORCH_CCL
 * TORCH_CCL_ENV_WAIT_GDB:          Default = 0, Set 1 to force the torch_ccl wait for GDB attaching
 */

#define TORCH_CCL_ENV_TYPE_DEF(var)                                     \
    int var = [&]() -> int {                                            \
      auto env = std::getenv("TORCH_CCL_" #var);                        \
      int _##var = 0;                                                   \
      try {                                                             \
        _##var = std::stoi(env, 0, 10);                                 \
      } catch (...) { /* Do Nothing */ }                                \
      return _##var;                                                    \
    } ()

int torch_ccl_env(int env_type) {

  static struct {
    TORCH_CCL_ENV_TYPE_DEF(ENV_VERBOSE);
    TORCH_CCL_ENV_TYPE_DEF(ENV_WAIT_GDB);
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
