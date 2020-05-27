#pragma once

#include <atomic>
#include <map>
#include <memory>

#include "ml_sst.hpp"
#include "multinomial_log_reg.hpp"
#include "tcp.hpp"

namespace coordinator {
class async_worker {
public:
    async_worker(log_reg::multinomial_log_reg& m_log_reg,
		 sst::MLSST& ml_sst, const uint32_t node_rank);

  void train(const size_t num_epochs);
  void train_SVRG(const size_t num_epochs);
  
private:
  log_reg::multinomial_log_reg& m_log_reg;
  sst::MLSST& ml_sst;
  const uint32_t node_rank;
};
}  // namespace coordinator
