
#include <sys/types.h>
#include <unistd.h>
#include "ccl_comm_collector.h"
#include "utils.h"


namespace oneccl_bindings_for_pytorch {

ccl::shared_ptr_class<ccl::kvs> CCLCommCollector::get_kvs(int rank, c10d::Store& store,
  bool singleP2POp = false, const std::string& p2pKey = "", int p2pRank = 0) {
  
  std::string storeKey;

  if (!singleP2POp) {
    storeKey = std::to_string(ccl_comms.size());
  } else {
    storeKey = p2pKey;
  }
  // Rank 0 broadcast the bootstrap network information to other ranks
  if (rank == 0 || (singleP2POp && p2pRank == 0)) {
    call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
        kvs = ccl::create_main_kvs();
    });
    ccl::kvs::address_type main_addr = kvs->get_address();
    auto ccl_kvs_addr = std::vector<uint8_t>(main_addr.begin(), main_addr.end());
    store.set(storeKey, ccl_kvs_addr);
  }
  else {
    auto ccl_kvs_addr = store.get(storeKey);
    if (ccl_kvs_addr.size() != ccl::kvs::address_max_size) {
      throw std::runtime_error(
              "Unexpected ccl kvs addr from the store\n");
    }
    ccl::kvs::address_type main_addr;
    std::copy_n(std::make_move_iterator(ccl_kvs_addr.begin()),
                ccl::kvs::address_max_size,
                main_addr.begin());
    call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
        kvs = ccl::create_kvs(main_addr);
    });
  }

  return kvs;
}

std::shared_ptr<oneccl_bindings_for_pytorch::Comms> CCLCommCollector::get_comms(const std::string& devices_key) {
  if (ccl_comms.find(devices_key) != ccl_comms.end()) {
    // Reuse the cached communicator if there is one.
    return ccl_comms[devices_key];
  }
  return {nullptr};
}

void CCLCommCollector::add_comms(const std::string& devices_key,
                                 std::shared_ptr<oneccl_bindings_for_pytorch::Comms> comms) {
  if (ccl_comms.find(devices_key) != ccl_comms.end()) {
    // Replace the cached comms
    ccl_comms[devices_key] = comms;
  } else {
    ccl_comms.emplace(devices_key, comms);
  }
}

}