#include "async_worker.hpp"

#include <chrono>
#include <map>
#include <sys/time.h>
#include <thread>

coordinator::async_worker::async_worker(log_reg::multinomial_log_reg& m_log_reg,
					sst::MLSST& ml_sst, const uint32_t node_rank)
  : m_log_reg(m_log_reg), ml_sst(ml_sst), node_rank(node_rank) {
}

void coordinator::async_worker::train(const size_t num_epochs) {
  std::cout << "4" << std::endl;
  ml_sst.sync_with_members();
  std::cout << "5" << std::endl;
  const size_t num_batches = m_log_reg.get_num_batches();
  uint64_t round = 0;
  const uint32_t num_grad_bufs = 3;

  for(size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
    for(size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
      std::cout << "round " << round
		<< " ml_sst.last_round " << ml_sst.last_round[0][node_rank] << std::endl;
      while (round >= ml_sst.last_round[0][node_rank] + num_grad_bufs) {
      }
      if (round % num_grad_bufs == 0) {
	m_log_reg.compute_gradient(batch_num, m_log_reg.get_model(), (double*)std::addressof(ml_sst.model_or_gradient[1][0]));
	ml_sst.round[1] = ++round;
	ml_sst.put_with_completion();
      } else if (round % num_grad_bufs == 1) {
	m_log_reg.compute_gradient(batch_num, m_log_reg.get_model(), (double*)std::addressof(ml_sst.model_or_gradient[1][1]));
	ml_sst.round1[1] = ++round;
	ml_sst.put_with_completion();
      } else {
	m_log_reg.compute_gradient(batch_num, m_log_reg.get_model(), (double*)std::addressof(ml_sst.model_or_gradient[1][2]));
	ml_sst.round2[1] = ++round;
	ml_sst.put_with_completion();
      }
    }
    ml_sst.sync_with_members();
    ml_sst.sync_with_members();
  }
  ml_sst.sync_with_members();
}

void coordinator::async_worker::train_SVRG(const size_t num_epochs) {
    const size_t num_batches = m_log_reg.get_num_batches();
    for(size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
        m_log_reg.copy_model(m_log_reg.get_model(),
  			   m_log_reg.get_anchor_model(),
			   m_log_reg.get_model_size());
        m_log_reg.compute_full_gradient(m_log_reg.get_anchor_model());

        for(size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
            m_log_reg.compute_gradient(batch_num,
				       m_log_reg.get_model());
	    m_log_reg.update_gradient(batch_num);
	    ml_sst.round[1]++;
	    ml_sst.put_with_completion();
        }
    }
    ml_sst.sync_with_members(); // for time_taken
}
