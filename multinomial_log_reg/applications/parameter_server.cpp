#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <sys/time.h>
#include <thread>
#include <vector>

#include "coordinator/ml_sst.hpp"
#include "coordinator/tcp.hpp"
#include "coordinator/verbs.h"
#include "multinomial_log_reg.hpp"
#include "utils/numpy_reader.hpp"
#include "utils/ml_stat.hpp"

int main(int argc, char* argv[]) {
    if(argc < 11) {
        std::cerr << "Usage: " << argv[0]
                  << " <data_directory> <syn/mnist/rff> <SVRG 0/1> <alpha> <decay> <aggregate_batch_size> <num_epochs> <num_nodes> <num_trials> <broadcast_completion_period>"
                  << std::endl;
        return 1;
    }
    std::string data_directory(argv[1]);
    std::string data(argv[2]);
    bool svrg = bool(atoi(argv[3]));

    const double gamma = 0.0001;
    const double alpha = std::stod(argv[4]);
    double decay = std::stod(argv[5]);
    if(svrg) {
        decay = 1.0;
    }
    uint32_t aggregate_batch_size = std::stod(argv[6]);
    const uint32_t num_epochs = atoi(argv[7]);
    const uint32_t num_nodes = atoi(argv[8]);
    std::string worker_dir = std::to_string(num_nodes - 1) + "workers";
    const uint32_t num_inner_epochs = 2;
    const uint32_t num_trials = atoi(argv[9]);
    const uint32_t broadcast_completion_period = atoi(argv[10]);
    const uint32_t num_grad_bufs = 3;

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
    for(uint32_t i = 0; i < num_nodes; ++i) {
        ip_addrs[i] = ip_addrs_static.at(i);
    }
    sst::verbs_initialize(ip_addrs, 0);
    std::vector<uint32_t> members(num_nodes);
    std::iota(members.begin(), members.end(), 0);

    const size_t batch_size = aggregate_batch_size / (num_nodes - 1);

    utils::ml_stats_t ml_stats(num_nodes, num_epochs);
    
    for(uint32_t trial_num = 0; trial_num < num_trials; ++trial_num) {
        log_reg::multinomial_log_reg m_log_reg(
                [&]() {
                    return (utils::dataset)numpy::numpy_dataset(
                            data_directory + "/" + data, (num_nodes - 1), 0);
                },
                alpha, gamma, decay, batch_size,
                svrg, num_inner_epochs);

        const size_t num_batches = m_log_reg.get_num_batches();

        sst::MLSST ml_sst(members, 0, m_log_reg.get_model_size());
        m_log_reg.set_model_mem((double*)std::addressof(
                ml_sst.model_or_gradient[0][0]));
	std::vector<uint64_t> num_lost_gradients(num_nodes, 0);
        utils::ml_stat_t ml_stat(trial_num, num_nodes, num_epochs,
				 alpha, decay, batch_size,
				 ml_sst, num_lost_gradients, m_log_reg);
	
	volatile bool trial_finished = false;
	// std::atomic<uint64_t> num_broadcasts = 0;
	uint64_t num_broadcasts = 0;
	for (uint row = 1; row < num_nodes; ++row) {
	  std::vector<double*> gradients;
	  gradients.push_back((double*)std::addressof(ml_sst.model_or_gradient[row][0]));
	  gradients.push_back((double*)std::addressof(ml_sst.model_or_gradient1[row][0]));
	  gradients.push_back((double*)std::addressof(ml_sst.model_or_gradient2[row][0]));
	  m_log_reg.push_back_to_grads_vec(gradients);
	}
	auto model_update_broadcast_loop = [&ml_sst, &trial_finished, num_nodes,
					    &num_epochs, &num_batches, num_grad_bufs,
					    &num_broadcasts,
					    &broadcast_completion_period,
					    &num_lost_gradients,
					    &m_log_reg]() mutable {
					     pthread_setname_np(pthread_self(), ("model_update_broadcast_loop"));
					     while(!trial_finished) {
					       std::vector<uint32_t> target_nodes;
					       for(uint row = 1; row < num_nodes; ++row) {
						 uint64_t round_max = std::max({ml_sst.round[row], ml_sst.round1[row], ml_sst.round2[row]});
						 std::cout << "row " << row
							   << " round " << ml_sst.round[row]
							   << " round1 " << ml_sst.round1[row]
							   << " round2 " << ml_sst.round2[row]
							   << " round_max " << round_max
							   << " last_round " << ml_sst.last_round[0][row] << std::endl;
						 if(ml_sst.last_round[0][row] < num_epochs * num_batches &&
						    round_max > ml_sst.last_round[0][row]) {
						     // std::cout << "Detected worker " << row << " pushed "
						     // << ml_sst.round[row] << " gradient." << std::endl;
						     for(uint i = ml_sst.last_round[0][row]; i < round_max; i++) {
						       uint col = ml_sst.last_round[0][row] % num_grad_bufs;
						       m_log_reg.update_model(row, col);
						       ml_sst.last_round[0][row]++;
						     }
						     if(round_max - ml_sst.last_round[0][row] > 0) {
						       num_lost_gradients[row] += round_max - ml_sst.last_round[0][row];
						     }
						     target_nodes.push_back(row);
						     ml_sst.round[0]++;
						 }
					       }
					       ml_sst.put_with_completion(target_nodes); // [TO FIX] use only the first col.
					       num_broadcasts++;
					     }
					   };
	std::thread model_update_broadcast_thread = std::thread(model_update_broadcast_loop);
        ml_sst.sync_with_members();

        struct timespec start_time, end_time;
        for(size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
	  clock_gettime(CLOCK_REALTIME, &start_time);
	  ml_sst.sync_with_members();
	  clock_gettime(CLOCK_REALTIME, &end_time);
	  double time_taken = (double)(end_time.tv_sec - start_time.tv_sec)
	    + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
	  ml_stat.initialize_epoch_parameters(epoch_num + 1, m_log_reg.get_model(),
					      ml_sst, num_broadcasts,
					      num_lost_gradients, time_taken);
	  ml_sst.sync_with_members();
        }
	
	trial_finished = true;
	model_update_broadcast_thread.join();
	std::cout << "trial_num " << trial_num << " done." << std::endl;
	std::cout << "Collecting results..." << std::endl;

	for(size_t epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
	  ml_stat.collect_results(epoch_num, m_log_reg);
	}
	ml_stats.push_back(ml_stat);

	ml_stat.print_results();
	ml_stat.fout_log_per_epoch();
	ml_stat.fout_analysis_per_epoch();
	ml_stat.fout_gradients_per_epoch();
        ml_sst.sync_with_members();

	if (trial_num == num_trials - 1) {
	  ml_stats.compute_mean();
	  ml_stats.compute_std();
	  ml_stats.compute_err();
    
	  // Write a file with hyper params and store model when cur_loss < prev_loss
	  std::string target_dir = data_directory + "/" + data + "/" + worker_dir;
	  ml_stats.grid_search_helper(target_dir, svrg);
	  ml_stats.fout_log_mean_per_epoch();
	  ml_stats.fout_log_err_per_epoch();
	  ml_stats.fout_analysis_mean_per_epoch();
	  ml_stats.fout_analysis_err_per_epoch();
	  ml_stats.fout_gradients_mean_per_epoch();
	  ml_stats.fout_gradients_err_per_epoch();
	  ml_sst.sync_with_members();
	}
    }
}
