#pragma once

#include "coordinator/ml_sst.hpp"
#include "multinomial_log_reg.hpp"

namespace utils {
class ml_stat_t {
public:
    ml_stat_t(uint32_t num_nodes, uint32_t num_epochs);
    ml_stat_t(uint32_t trial_num, uint32_t num_nodes,
	       uint32_t num_epochs, double alpha,
	      double decay, double batch_size,
              const sst::MLSST& ml_sst,
	      std::vector<uint64_t>& num_lost_gradients,
              log_reg::multinomial_log_reg& m_log_reg);

    void initialize_epoch_parameters(
            uint32_t epoch_num, double* model,
            const sst::MLSST& ml_sst, uint64_t num_broadcasts,
	    std::vector<uint64_t>& num_lost_gradients, double time_taken);
    void collect_results(uint32_t epoch_num, log_reg::multinomial_log_reg& m_log_reg);
    void print_results();
    void fout_log_per_epoch();
    void fout_analysis_per_epoch();
    void fout_gradients_per_epoch();

    uint32_t trial_num;
    uint32_t num_nodes;
    uint32_t num_epochs;
    double alpha;
    double decay;
    double batch_size;
    size_t model_size;

    std::vector<double*> intermediate_models;
    std::vector<double> cumulative_num_broadcasts;
    std::vector<double> num_model_updates;
    // The first row of num_gradients_received is used for the sum of each worker's num_pushed gradients.
    std::vector<std::vector<double>> num_gradients_received;
    std::vector<std::vector<double>> num_lost_gradients;
  
    std::vector<double> cumulative_time;
    std::vector<double> training_error;
    std::vector<double> test_error;
    std::vector<double> loss_gap;
    std::vector<double> dist_to_opt;
    std::vector<double> grad_norm;
};

class ml_stats_t {
public:
  ml_stats_t(uint32_t num_nodes, uint32_t num_epochs);
  void push_back(ml_stat_t ml_stat);
  void compute_mean();
  void compute_std();
  void compute_err();
  void grid_search_helper(std::string target_dir, bool svrg);
  void fout_log_mean_per_epoch();
  void fout_log_err_per_epoch();
  void fout_analysis_mean_per_epoch();
  void fout_analysis_err_per_epoch();
  void fout_gradients_mean_per_epoch();
  void fout_gradients_err_per_epoch();
  
  std::vector<ml_stat_t> ml_stat_vec;
  ml_stat_t mean;
  ml_stat_t std;
  ml_stat_t err;
  
private:
  double get_loss_opt(std::string target_dir);
};
}  // namespace utils
