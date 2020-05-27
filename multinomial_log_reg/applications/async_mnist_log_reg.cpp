#include <fstream>
#include <iostream>
#include <utility>
#include <unistd.h>
#include <queue>

#include "coordinator/async_worker.hpp"
#include "utils/numpy_reader.hpp"

int main(int argc, char* argv[]) {
    if(argc < 12) {
        std::cerr << "Usage: " << argv[0]
		  << " <data_directory> <syn/mnist/rff> <SVRG 0/1> <alpha> <decay> <aggregate_batch_size> <num_epochs> <node_rank> <num_nodes> <num_trials> <grad_push_completion_period>"
                  << std::endl;
        return 1;
    }
    std::string data_directory(argv[1]);
    std::string data(argv[2]);
    bool svrg = bool(atoi(argv[3]));

    const double gamma = 0.0001;
    const double alpha = std::stod(argv[4]);
    double decay = std::stod(argv[5]);
    if (svrg) {
      decay = 1.0;
    }
    uint32_t aggregate_batch_size = std::stod(argv[6]);
    const uint32_t num_epochs = atoi(argv[7]);
    const uint32_t node_rank = atoi(argv[8]);
    const uint32_t num_nodes = atoi(argv[9]);
    const uint32_t num_inner_epochs = 2;
    const uint32_t num_trials = atoi(argv[10]);
    const uint32_t grad_push_completion_period = atoi(argv[11]);
    const uint32_t num_grad_push_threads = 16;
    
    std::map<uint32_t, std::string> ip_addrs_static;
    ip_addrs_static[0] = "192.168.99.16";
    ip_addrs_static[1] = "192.168.99.17";
    ip_addrs_static[2] = "192.168.99.18";
    ip_addrs_static[3] = "192.168.99.20";
    ip_addrs_static[4] = "192.168.99.30";
    ip_addrs_static[5] = "192.168.99.31";
    ip_addrs_static[6] = "192.168.99.32";
    ip_addrs_static[7] = "192.168.99.24";
    ip_addrs_static[8] = "192.168.99.25";
    ip_addrs_static[9] = "192.168.99.27";
    ip_addrs_static[10] = "192.168.99.23";
    ip_addrs_static[11] = "192.168.99.105";
    ip_addrs_static[12] = "192.168.99.29";
    ip_addrs_static[13] = "192.168.99.26";
    ip_addrs_static[14] = "192.168.99.106";
    ip_addrs_static[15] = "192.168.99.28";

    std::map<uint32_t, std::string> ip_addrs;
    ip_addrs[0] = ip_addrs_static.at(0);
    ip_addrs[node_rank] = ip_addrs_static.at(node_rank);
    sst::verbs_initialize(ip_addrs, node_rank);
    
    const size_t batch_size = aggregate_batch_size / (num_nodes - 1);

    for(uint32_t trial_num = 0; trial_num < num_trials; ++trial_num) {
      std::cout << "trial_num " << trial_num << std::endl;
      log_reg::multinomial_log_reg m_log_reg(
					     [&]() {
					       return (utils::dataset)numpy::numpy_dataset(
											   data_directory + "/" + data, (num_nodes - 1), node_rank - 1);
					     },
					     alpha, gamma, decay, batch_size,
					     svrg, num_inner_epochs);
      std::cout << "1" << std::endl;
      sst::MLSST ml_sst(std::vector<uint32_t>{0, node_rank},
					      node_rank, m_log_reg.get_model_size());

      m_log_reg.set_model_mem((double*)std::addressof(ml_sst.model_or_gradient[0][0]));
      std::vector<double*> gradients;
      gradients.push_back((double*)std::addressof(ml_sst.model_or_gradient[1][0]));
      gradients.push_back((double*)std::addressof(ml_sst.model_or_gradient1[1][0]));
      gradients.push_back((double*)std::addressof(ml_sst.model_or_gradient2[1][0]));
      m_log_reg.push_back_to_grads_vec(gradients);
      std::cout << "2" << std::endl;
      
      coordinator::async_worker worker(m_log_reg, ml_sst, node_rank);
      if (svrg) {
	worker.train_SVRG(num_epochs);
      } else {
	worker.train(num_epochs);
      }
      std::cout << "3" << std::endl;
      
      if (trial_num == num_trials - 1) {
	ml_sst.sync_with_members();
      }
    }
}
