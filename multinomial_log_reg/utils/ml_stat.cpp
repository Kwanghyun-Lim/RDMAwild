#include <assert.h>
#include <cmath>
#include <fstream>
#include <string>

#include "ml_stat.hpp"

utils::ml_stat_t::ml_stat_t(uint32_t num_nodes, uint32_t num_epochs) :
  trial_num(0),
  num_nodes(0),
  num_epochs(0),
  alpha(0.0),
  decay(0.0),
  batch_size(0),
  model_size(0),
  intermediate_models(num_epochs + 1),
  cumulative_num_broadcasts(num_epochs + 1, 0),
  num_model_updates(num_epochs + 1, 0),
  num_gradients_received(num_epochs + 1, std::vector<double>(num_nodes, 0)),
  num_lost_gradients(num_epochs + 1, std::vector<double>(num_nodes, 0)),
  cumulative_time(num_epochs + 1, 0),
  training_error(num_epochs + 1, 0),
  test_error(num_epochs + 1, 0),
  loss_gap(num_epochs + 1, 0),
  dist_to_opt(num_epochs + 1, 0),
  grad_norm(num_epochs + 1, 0) {
}

utils::ml_stat_t::ml_stat_t(uint32_t trial_num, uint32_t num_nodes,
	       uint32_t num_epochs, double alpha,
	      double decay, double batch_size,
              const sst::MLSST& ml_sst,
	      std::vector<uint64_t>& num_lost_gradients,
              log_reg::multinomial_log_reg& m_log_reg)
        : trial_num(trial_num),
          num_nodes(num_nodes),
          num_epochs(num_epochs),
          alpha(alpha),
          decay(decay),
          batch_size(batch_size),
          model_size(m_log_reg.get_model_size()),
          intermediate_models(num_epochs + 1),
          cumulative_num_broadcasts(num_epochs + 1, 0),
          num_model_updates(num_epochs + 1, 0),
          num_gradients_received(num_epochs + 1, std::vector<double>(num_nodes, 0)),
	  num_lost_gradients(num_epochs + 1, std::vector<double>(num_nodes, 0)),
          cumulative_time(num_epochs + 1, 0),
          training_error(num_epochs + 1, 0),
          test_error(num_epochs + 1, 0),
          loss_gap(num_epochs + 1, 0),
          dist_to_opt(num_epochs + 1, 0),
          grad_norm(num_epochs + 1, 0) {
    for(uint epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
        intermediate_models[epoch_num] = new double[model_size];
        for(uint i = 0; i < model_size; ++i) {
            intermediate_models[epoch_num][model_size] = 0;
        }
    }

    initialize_epoch_parameters(0, m_log_reg.get_model(),
				ml_sst, 0,
				num_lost_gradients, 0);
}

void utils::ml_stat_t::initialize_epoch_parameters(
        uint32_t epoch_num, double* model,
        const sst::MLSST& ml_sst, uint64_t num_broadcasts,
	std::vector<uint64_t>& num_lost_gradients, double time_taken) {
    for(size_t i = 0; i < model_size; ++i) {
        intermediate_models[epoch_num][i] = model[i];
    }
    if (epoch_num == 0) {
      cumulative_time[epoch_num] = time_taken;
    } else {
      cumulative_time[epoch_num] = cumulative_time[epoch_num-1] + time_taken;
    }
    cumulative_num_broadcasts[epoch_num] = num_broadcasts;
    num_model_updates[epoch_num] = ml_sst.round[0];
    for(uint node_num = 1; node_num < num_nodes; ++node_num) {
        num_gradients_received[epoch_num][node_num] = ml_sst.round[node_num];
        num_gradients_received[epoch_num][0] += num_gradients_received[epoch_num][node_num];
	this->num_lost_gradients[epoch_num][node_num] = (double)num_lost_gradients[node_num];
	this->num_lost_gradients[epoch_num][0] += this->num_lost_gradients[epoch_num][node_num];
    }
}

void utils::ml_stat_t::collect_results(uint32_t epoch_num, log_reg::multinomial_log_reg& m_log_reg) {
    m_log_reg.set_model_mem(intermediate_models[epoch_num]);
    training_error[epoch_num] = m_log_reg.training_error();
    test_error[epoch_num] = m_log_reg.test_error();
    loss_gap[epoch_num] = m_log_reg.training_loss() - m_log_reg.get_loss_opt();
    dist_to_opt[epoch_num] = m_log_reg.distance_to_optimum();
    grad_norm[epoch_num] = m_log_reg.gradient_norm();
}

void utils::ml_stat_t::print_results() {
  std::cout << "trial_num " << trial_num << std::endl;
  std::cout << "num_workers " << num_nodes - 1 << std::endl;
  std::cout << "num_epochs " << num_epochs << std::endl;
  std::cout << "alpha " << alpha << std::endl;
  std::cout << "decay " << decay << std::endl;
  std::cout << "batch_size " << batch_size << std::endl;
  std::cout << "time_taken " << cumulative_time[num_epochs] << "s" << std::endl;
  std::cout << "training_accuracy " << 100 * (1 - training_error[num_epochs]) << std::endl;
  std::cout << "test_accuracy " << 100 * (1 - test_error[num_epochs]) << std::endl;
  std::cout << "loss_gap " << loss_gap[num_epochs] << std::endl;
  std::cout << "dist_to_opt " << dist_to_opt[num_epochs] << std::endl;
  std::cout << "grad_norm " << grad_norm[num_epochs] << std::endl;
  std::cout << std::endl;
}

void utils::ml_stat_t::fout_log_per_epoch() {
  std::ofstream log_file("RDMAwild.log", std::ofstream::app);
  log_file << "trial_num "
	   << "num_workers "
	   << "num_epochs "
	   << "alpha "
	   << "decay "
	   << "batch_size "
	   << "time_taken "
	   << "training_accuracy "
	   << "test_accuracy "
	   << "loss_gap "
	   << "dist_to_opt "
	   << "grad_norm "
	   << std::endl;

  for(uint32_t epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
    log_file << trial_num << " "
	     << num_nodes - 1 << " "
	     << epoch_num << " "
	     << alpha << " "
	     << decay << " "
	     << batch_size << " "
	     << cumulative_time[epoch_num] << " "
	     << 100 * (1 - training_error[epoch_num]) << " "
	     << 100 * (1 - test_error[epoch_num]) << " "
	     << loss_gap[epoch_num] << " "
	     << dist_to_opt[epoch_num] << " "
	     << grad_norm[epoch_num] << " "
	     << std::endl;
  }
}

void utils::ml_stat_t::fout_analysis_per_epoch() {
  std::ofstream analysis_file("RDMAwild.analysis", std::ofstream::app);
  analysis_file << "trial_num "
		<< "num_workers "
		<< "num_epochs "
		<< "alpha "
		<< "decay "
		<< "batch_size "
		<< "num_broadcasts "
		<< "num_model_updates "
		<< "num_lost_gradients "
		<< "num_gradients_received "
		<< std::endl;

  for (uint32_t epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
    analysis_file << trial_num << " "
		  << num_nodes - 1 << " "
		  << epoch_num << " "
		  << alpha << " "
		  << decay << " "
		  << batch_size << " "
		  << cumulative_num_broadcasts[epoch_num] << " "
		  << num_model_updates[epoch_num] << " "
		  << num_lost_gradients[epoch_num][0] << " "
		  << num_gradients_received[epoch_num][0] << " "
		  << std::endl;
  }
}

void utils::ml_stat_t::fout_gradients_per_epoch() {
  std::ofstream gradients_file("RDMAwild.gradients", std::ofstream::app);
  gradients_file << "trial_num "
		 << "num_workers "
		 << "alpha "
		 << "decay "
		 << "batch_size "
		 << "num_epochs ";
  for (uint node_num = 0; node_num < num_nodes; ++node_num) {
    if (node_num == 0) {
      gradients_file << "num_lost_gradients_sum ";
    }
    gradients_file << "node" << node_num << " ";
  }
  for (uint node_num = 0; node_num < num_nodes; ++node_num) {
    if (node_num == 0) {
      gradients_file << "num_gradients_received_sum ";
    }
    gradients_file << "node" << node_num << " ";
  }

  gradients_file << std::endl;
    
  for (uint32_t epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
    gradients_file << trial_num << " "
		   << num_nodes - 1 << " "
		   << alpha << " "
		   << decay << " "
		   << batch_size << " "
		   << epoch_num << " ";
    for (uint node_num = 0; node_num < num_nodes; ++node_num) {
      gradients_file << num_lost_gradients[epoch_num][node_num] << " ";
    }
    for (uint node_num = 0; node_num < num_nodes; ++node_num) {
      gradients_file << num_gradients_received[epoch_num][node_num] << " ";
    }
    gradients_file << std::endl;
  }
}

utils::ml_stats_t::ml_stats_t(uint32_t num_nodes, uint32_t num_epochs)
  : ml_stat_vec(), mean(num_nodes, num_epochs)
    , std(num_nodes, num_epochs), err(num_nodes, num_epochs) {
}

void utils::ml_stats_t::push_back(ml_stat_t ml_stat) {
  ml_stat_vec.push_back(ml_stat);
}

void utils::ml_stats_t::compute_mean() {
  assert (ml_stat_vec.size() > 0);
  uint num_epochs = ml_stat_vec[0].num_epochs;
  uint num_trials = ml_stat_vec.size();
  uint num_nodes = ml_stat_vec[0].num_nodes;
    
  for (uint epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
    for (uint trial_num = 0; trial_num < num_trials; ++trial_num) {
      mean.cumulative_time[epoch_num] += ml_stat_vec[trial_num].cumulative_time[epoch_num] / num_trials;
      mean.training_error[epoch_num] += ml_stat_vec[trial_num].training_error[epoch_num] / num_trials;
      mean.test_error[epoch_num] += ml_stat_vec[trial_num].test_error[epoch_num] / num_trials;
      mean.loss_gap[epoch_num] += ml_stat_vec[trial_num].loss_gap[epoch_num] / num_trials;
      mean.dist_to_opt[epoch_num] += ml_stat_vec[trial_num].dist_to_opt[epoch_num] / num_trials;
      mean.grad_norm[epoch_num] += ml_stat_vec[trial_num].grad_norm[epoch_num] / num_trials;

      mean.cumulative_num_broadcasts[epoch_num] += ml_stat_vec[trial_num].cumulative_num_broadcasts[epoch_num] / num_trials;
      mean.num_model_updates[epoch_num] += ml_stat_vec[trial_num].num_model_updates[epoch_num] / num_trials;

      for (uint node_num = 0; node_num < num_nodes; ++node_num) {
      	mean.num_gradients_received[epoch_num][node_num] += ml_stat_vec[trial_num].num_gradients_received[epoch_num][node_num] / num_trials;
      	mean.num_lost_gradients[epoch_num][node_num] += ml_stat_vec[trial_num].num_lost_gradients[epoch_num][node_num] / num_trials;
      }
    }
  }
}

void utils::ml_stats_t::compute_std() {
  assert (ml_stat_vec.size() > 0);
  uint num_epochs = ml_stat_vec[0].num_epochs;
  uint num_trials = ml_stat_vec.size();
  uint num_nodes = ml_stat_vec[0].num_nodes;

  for (uint epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
    for (uint trial_num = 0; trial_num < num_trials; ++trial_num) {
  	std.cumulative_time[epoch_num]
  	  += std::pow((ml_stat_vec[trial_num].cumulative_time[epoch_num] - mean.cumulative_time[epoch_num]), 2);
  	std.training_error[epoch_num]
  	  += std::pow((ml_stat_vec[trial_num].training_error[epoch_num] - mean.training_error[epoch_num]), 2);
  	std.test_error[epoch_num]
  	  += std::pow((ml_stat_vec[trial_num].test_error[epoch_num] - mean.test_error[epoch_num]), 2);
  	std.loss_gap[epoch_num]
  	  += std::pow((ml_stat_vec[trial_num].loss_gap[epoch_num] - mean.loss_gap[epoch_num]), 2);
  	std.dist_to_opt[epoch_num]
  	  += std::pow((ml_stat_vec[trial_num].dist_to_opt[epoch_num] - mean.dist_to_opt[epoch_num]), 2);
  	std.grad_norm[epoch_num]
  	  += std::pow((ml_stat_vec[trial_num].grad_norm[epoch_num] - mean.grad_norm[epoch_num]), 2);

  	std.cumulative_num_broadcasts[epoch_num]
  	  += std::pow((ml_stat_vec[trial_num].cumulative_num_broadcasts[epoch_num] - mean.cumulative_num_broadcasts[epoch_num]), 2);
  	std.num_model_updates[epoch_num]
  	  += std::pow((ml_stat_vec[trial_num].num_model_updates[epoch_num] - mean.num_model_updates[epoch_num]), 2);
  	for (uint node_num = 0; node_num < num_nodes; ++node_num) {
  	  std.num_gradients_received[epoch_num][node_num]
  	    += std::pow((ml_stat_vec[trial_num].num_gradients_received[epoch_num][node_num] - mean.num_gradients_received[epoch_num][node_num]), 2);
  	  std.num_lost_gradients[epoch_num][node_num]
  	    += std::pow((ml_stat_vec[trial_num].num_lost_gradients[epoch_num][node_num] - mean.num_lost_gradients[epoch_num][node_num]), 2);
  	}
    }
  }

  for (uint epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
    std.cumulative_time[epoch_num] = std::sqrt(std.cumulative_time[epoch_num] / num_trials);
    std.training_error[epoch_num] = std::sqrt(std.training_error[epoch_num] / num_trials);
    std.test_error[epoch_num] = std::sqrt(std.test_error[epoch_num] / num_trials);
    std.loss_gap[epoch_num] = std::sqrt(std.loss_gap[epoch_num] / num_trials);
    std.dist_to_opt[epoch_num] = std::sqrt(std.dist_to_opt[epoch_num] / num_trials);
    std.grad_norm[epoch_num] = std::sqrt(std.grad_norm[epoch_num] / num_trials);
    std.cumulative_num_broadcasts[epoch_num] = std::sqrt(std.cumulative_num_broadcasts[epoch_num] / num_trials);
    std.num_model_updates[epoch_num] = std::sqrt(std.num_model_updates[epoch_num] / num_trials);
    for (uint node_num = 0; node_num < num_nodes; ++node_num) {
      std.num_gradients_received[epoch_num][node_num] = std::sqrt(std.num_gradients_received[epoch_num][node_num] / num_trials);
      std.num_lost_gradients[epoch_num][node_num] = std::sqrt(std.num_lost_gradients[epoch_num][node_num] / num_trials);
    }
  }
}

void utils::ml_stats_t::compute_err() {
  assert (ml_stat_vec.size() > 0);
  uint num_epochs = ml_stat_vec[0].num_epochs;
  uint num_trials = ml_stat_vec.size();
  uint num_nodes = ml_stat_vec[0].num_nodes;
  
  // 95% confidence error
  double confidence_num = 1.960;
  for (uint epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
    err.cumulative_time[epoch_num] = confidence_num * std.cumulative_time[epoch_num] / std::sqrt(num_trials);
    err.training_error[epoch_num] = confidence_num * std.training_error[epoch_num] / std::sqrt(num_trials);
    err.test_error[epoch_num] = confidence_num * std.test_error[epoch_num] / std::sqrt(num_trials);
    err.loss_gap[epoch_num] = confidence_num * std.loss_gap[epoch_num] / std::sqrt(num_trials);
    err.dist_to_opt[epoch_num] = confidence_num * std.dist_to_opt[epoch_num] / std::sqrt(num_trials);
    err.grad_norm[epoch_num] = confidence_num * std.grad_norm[epoch_num] / std::sqrt(num_trials);
    err.cumulative_num_broadcasts[epoch_num] = confidence_num * std.cumulative_num_broadcasts[epoch_num] / std::sqrt(num_trials);
    err.num_model_updates[epoch_num] = confidence_num * std.num_model_updates[epoch_num] / std::sqrt(num_trials);
    for (uint node_num = 0; node_num < num_nodes; ++node_num) {
      err.num_gradients_received[epoch_num][node_num] = confidence_num * std.num_gradients_received[epoch_num][node_num] / std::sqrt(num_trials);
      err.num_lost_gradients[epoch_num][node_num] = confidence_num * std.num_lost_gradients[epoch_num][node_num] / std::sqrt(num_trials);
    }
  }
}
  
void utils::ml_stats_t::grid_search_helper(std::string target_dir, bool svrg) {
    assert (ml_stat_vec.size() > 0);
    uint num_epochs = ml_stat_vec[0].num_epochs;
    std::ifstream prev_loss_opt_file;
    uint num_nodes = ml_stat_vec[0].num_nodes;
    uint batch_size = ml_stat_vec[0].batch_size;
    double alpha = ml_stat_vec[0].alpha;
    double decay = ml_stat_vec[0].decay;
    
    if (svrg) {
      prev_loss_opt_file.open(
	   target_dir + "/svrg_loss_opt.txt");
    } else {
      prev_loss_opt_file.open(
	   target_dir + "/sgd_loss_opt.txt");
    }
    
    std::string prev_loss_opt_str;
    prev_loss_opt_file >> prev_loss_opt_str;
    const double prev_loss_opt = std::stod(prev_loss_opt_str);
    prev_loss_opt_file.close();
    double cur_loss = mean.loss_gap[num_epochs] + get_loss_opt(target_dir);
    uint32_t aggregate_batch_size = batch_size * num_nodes;
    if (cur_loss < prev_loss_opt) {
      std::cout << "cur_loss " << cur_loss
		<< " < prev_loss_opt " << prev_loss_opt << std::endl;
      std::cout << "Found better alpha " << alpha << " decay " << decay
		<< " and aggregate_batch_size " << aggregate_batch_size << std::endl;
      
      if (svrg) {
    	std::ofstream svrg_alpha_opt_file(target_dir + "/svrg_alpha_opt.txt");
    	svrg_alpha_opt_file << alpha;
    	std::ofstream svrg_batch_opt_file(target_dir + "/svrg_batch_opt.txt");
    	svrg_batch_opt_file << aggregate_batch_size;
    	std::ofstream svrg_epoch_opt_file(target_dir + "/svrg_epoch_opt.txt");
    	svrg_epoch_opt_file << num_epochs;
    	std::ofstream svrg_loss_opt_file(target_dir + "/svrg_loss_opt.txt");
    	svrg_loss_opt_file << cur_loss;
      } else {
    	std::ofstream sgd_alpha_opt_file(target_dir + "/sgd_alpha_opt.txt");
    	sgd_alpha_opt_file << alpha;
    	std::ofstream sgd_decay_opt_file(target_dir + "/sgd_decay_opt.txt");
    	sgd_decay_opt_file << decay;
    	std::ofstream sgd_batch_opt_file(target_dir + "/sgd_batch_opt.txt");
    	sgd_batch_opt_file << aggregate_batch_size;
    	std::ofstream sgd_epoch_opt_file(target_dir + "/sgd_epoch_opt.txt");
    	sgd_epoch_opt_file << num_epochs;
    	std::ofstream sgd_loss_opt_file(target_dir + "/sgd_loss_opt.txt");
    	sgd_loss_opt_file << cur_loss;
      }
    } else {
      std::cout << "cur_loss " << cur_loss
		<< " > prev_loss_opt " << prev_loss_opt << std::endl;
      std::cout << "This alpha " << alpha << " decay " << decay
		<< " and aggregate_batch_size " << aggregate_batch_size
    		<< " is NOT better than the current ones." << std::endl;
    }
}

double utils::ml_stats_t::get_loss_opt(std::string target_dir) {
  std::ifstream loss_opt_file;
  loss_opt_file.open(target_dir + "/svrg_loss_opt.txt");
  std::string loss_opt_str;
  loss_opt_file >> loss_opt_str;
  double loss_opt = std::stod(loss_opt_str);
  return loss_opt;
}

void utils::ml_stats_t::fout_log_mean_per_epoch() {
  assert (ml_stat_vec.size() > 0);
  uint num_epochs = ml_stat_vec[0].num_epochs;
  uint num_nodes = ml_stat_vec[0].num_nodes;
  uint num_trials = ml_stat_vec.size();
  uint batch_size = ml_stat_vec[0].batch_size;
  double alpha = ml_stat_vec[0].alpha;
  double decay = ml_stat_vec[0].decay;

  std::ofstream mean_file("RDMAwild.log.mean", std::ofstream::app);
  mean_file << "num_trials "
	    << "num_workers "
	    << "num_epochs "
	    << "alpha "
	    << "decay "
	    << "batch_size "
	    << "time_taken "
	    << "training_accuracy "
	    << "test_accuracy "
	    << "loss_gap "
	    << "dist_to_opt "
	    << "grad_norm "
	    << std::endl;

  for (uint32_t epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
    mean_file << num_trials << " "
	      << num_nodes - 1 << " "
	      << epoch_num << " "
	      << alpha << " "
	      << decay << " "
	      << batch_size << " "
	      << mean.cumulative_time[epoch_num] << " "
	      << 100 * (1 - mean.training_error[epoch_num]) << " "
	      << 100 * (1 - mean.test_error[epoch_num]) << " "
	      << mean.loss_gap[epoch_num] << " "
	      << mean.dist_to_opt[epoch_num] << " "
	      << mean.grad_norm[epoch_num] << " "
	      << std::endl;
  }
}

void utils::ml_stats_t::fout_log_err_per_epoch() {
  assert (ml_stat_vec.size() > 0);
  uint num_epochs = ml_stat_vec[0].num_epochs;
  uint num_nodes = ml_stat_vec[0].num_nodes;
  uint num_trials = ml_stat_vec.size();
  uint batch_size = ml_stat_vec[0].batch_size;
  double alpha = ml_stat_vec[0].alpha;
  double decay = ml_stat_vec[0].decay;
  
  std::ofstream err_file("RDMAwild.log.err", std::ofstream::app);
  err_file << "num_trials "
	   << "num_workers "
	   << "num_epochs "
	   << "alpha "
	   << "decay "
	   << "batch_size "
	   << "time_taken "
	   << "training_accuracy "
	   << "test_accuracy "
	   << "loss_gap "
	   << "dist_to_opt "
	   << "grad_norm "
	   << std::endl;

  for (uint32_t epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
    err_file << num_trials << " "
  	   << num_nodes - 1 << " "
  	   << epoch_num << " "
  	   << alpha << " "
  	   << decay << " "
  	   << batch_size << " "
  	   << err.cumulative_time[epoch_num] << " "
  	   << 100 * err.training_error[epoch_num] << " "
  	   << 100 * err.test_error[epoch_num] << " "
  	   << err.loss_gap[epoch_num] << " "
  	   << err.dist_to_opt[epoch_num] << " "
  	   << err.grad_norm[epoch_num] << " "
  	   << std::endl;
  }
}

void utils::ml_stats_t::fout_analysis_mean_per_epoch() {
  assert (ml_stat_vec.size() > 0);
  uint num_trials = ml_stat_vec.size();
  uint num_nodes = ml_stat_vec[0].num_nodes;
  uint num_epochs = ml_stat_vec[0].num_epochs;
  double alpha = ml_stat_vec[0].alpha;
  double decay = ml_stat_vec[0].decay;
  uint batch_size = ml_stat_vec[0].batch_size;

  std::ofstream analysis_mean_file("RDMAwild.analysis.mean", std::ofstream::app);
  analysis_mean_file << "num_trials "
		     << "num_workers "
		     << "num_epochs "
		     << "alpha "
		     << "decay "
		     << "batch_size "
		     << "num_broadcasts "
		     << "num_model_updates "
		     << "num_lost_gradients "
		     << "num_gradients_received "
		     << std::endl;

  for (uint32_t epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
    analysis_mean_file << num_trials << " "
		       << num_nodes - 1 << " "
		       << epoch_num << " "
		       << alpha << " "
		       << decay << " "
		       << batch_size << " "
		       << mean.cumulative_num_broadcasts[epoch_num] << " "
		       << mean.num_model_updates[epoch_num] << " "
		       << mean.num_lost_gradients[epoch_num][0] << " "
		       << mean.num_gradients_received[epoch_num][0] << " "
		       << std::endl;
  }
}

void utils::ml_stats_t::fout_analysis_err_per_epoch() {
  assert (ml_stat_vec.size() > 0);
  uint num_trials = ml_stat_vec.size();
  uint num_nodes = ml_stat_vec[0].num_nodes;
  uint num_epochs = ml_stat_vec[0].num_epochs;
  double alpha = ml_stat_vec[0].alpha;
  double decay = ml_stat_vec[0].decay;
  uint batch_size = ml_stat_vec[0].batch_size;
  
  std::ofstream analysis_err_file("RDMAwild.analysis.err", std::ofstream::app);
  analysis_err_file << "num_trials "
		    << "num_workers "
		    << "num_epochs "
		    << "alpha "
		    << "decay "
		    << "batch_size "
		    << "num_broadcasts "
		    << "num_model_updates "
		    << "num_lost_gradients "
		    << "num_gradients_received "
		    << std::endl;

  for (uint32_t epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
    analysis_err_file << num_trials << " "
		      << num_nodes - 1 << " "
		      << epoch_num << " "
		      << alpha << " "
		      << decay << " "
		      << batch_size << " "
		      << err.cumulative_num_broadcasts[epoch_num] << " "
		      << err.num_model_updates[epoch_num] << " "
		      << err.num_lost_gradients[epoch_num][0] << " "
		      << err.num_gradients_received[epoch_num][0] << " "
		      << std::endl;
  }
}

void utils::ml_stats_t::fout_gradients_mean_per_epoch() {
  assert (ml_stat_vec.size() > 0);
  uint num_trials = ml_stat_vec.size();
  uint num_nodes = ml_stat_vec[0].num_nodes;
  uint num_epochs = ml_stat_vec[0].num_epochs;
  double alpha = ml_stat_vec[0].alpha;
  double decay = ml_stat_vec[0].decay;
  uint batch_size = ml_stat_vec[0].batch_size;

  std::ofstream gradients_mean_file("RDMAwild.gradients.mean", std::ofstream::app);
  gradients_mean_file << "num_trials "
		      << "num_workers "
		      << "alpha "
		      << "decay "
		      << "batch_size "
		      << "num_epochs ";
  for (uint node_num = 0; node_num < num_nodes; ++node_num) {
    if (node_num == 0) {
      gradients_mean_file << "num_lost_gradients_sum ";
    }
    gradients_mean_file << "node" << node_num << " ";
  }
  for (uint node_num = 0; node_num < num_nodes; ++node_num) {
    if (node_num == 0) {
      gradients_mean_file << "num_gradients_received_sum ";
    }
    gradients_mean_file << "node" << node_num << " ";
  }
  gradients_mean_file << std::endl;
    
  for (uint32_t epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
    gradients_mean_file << num_trials << " "
			<< num_nodes - 1 << " "
			<< alpha << " "
			<< decay << " "
			<< batch_size << " "
			<< epoch_num << " ";
    for (uint node_num = 0; node_num < num_nodes; ++node_num) {
      gradients_mean_file << mean.num_lost_gradients[epoch_num][node_num] << " ";
    }
    for (uint node_num = 0; node_num < num_nodes; ++node_num) {
      gradients_mean_file << mean.num_gradients_received[epoch_num][node_num] << " ";
    }
    gradients_mean_file << std::endl;
  }
}

void utils::ml_stats_t::fout_gradients_err_per_epoch() {
  assert (ml_stat_vec.size() > 0);
  uint num_trials = ml_stat_vec.size();
  uint num_nodes = ml_stat_vec[0].num_nodes;
  uint num_epochs = ml_stat_vec[0].num_epochs;
  double alpha = ml_stat_vec[0].alpha;
  double decay = ml_stat_vec[0].decay;
  uint batch_size = ml_stat_vec[0].batch_size;

  std::ofstream gradients_err_file("RDMAwild.gradients.err", std::ofstream::app);
  gradients_err_file << "num_trials "
  	 << "num_workers "
  	 << "alpha "
  	 << "decay "
  	 << "batch_size "
  	 << "num_epochs ";
  for (uint node_num = 0; node_num < num_nodes; ++node_num) {
    if (node_num == 0) {
  	gradients_err_file << "num_lost_gradients_sum ";
    }
    gradients_err_file << "node" << node_num << " ";
  }
  for (uint node_num = 0; node_num < num_nodes; ++node_num) {
    if (node_num == 0) {
  	gradients_err_file << "num_gradients_received_sum ";
    }
    gradients_err_file << "node" << node_num << " ";
  }
  gradients_err_file << std::endl;
    
  for (uint32_t epoch_num = 0; epoch_num < num_epochs + 1; ++epoch_num) {
    gradients_err_file << num_trials << " "
  	   << num_nodes - 1 << " "
  	   << alpha << " "
  	   << decay << " "
  	   << batch_size << " "
  	   << epoch_num << " ";
    for (uint node_num = 0; node_num < num_nodes; ++node_num) {
  	gradients_err_file << err.num_lost_gradients[epoch_num][node_num] << " ";
    }
    for (uint node_num = 0; node_num < num_nodes; ++node_num) {
  	gradients_err_file << err.num_gradients_received[epoch_num][node_num] << " ";
    }
    gradients_err_file << std::endl;
  }
}

