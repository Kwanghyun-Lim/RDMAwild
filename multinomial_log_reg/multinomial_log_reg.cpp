#include "multinomial_log_reg.hpp"

#include <assert.h>
#include <cblas.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

log_reg::multinomial_log_reg::multinomial_log_reg(
        const utils::reader_t& dataset_loader,
        const double alpha, const double gamma, double decay, const size_t batch_size,
	const bool svrg, const uint32_t num_inner_epochs)
        : dataset(dataset_loader()),
          model_size(dataset.training_labels.num_classes
                     * dataset.training_images.num_pixels),
	  gradients(),
	  svrg(svrg),
	  num_inner_epochs(num_inner_epochs),
	  full_gradient(new double[model_size]),
	  full_predicted_labels(std::make_unique<double[]>(
		           dataset.training_labels.num_classes * dataset.training_images.num_total_images)),
          alpha(alpha / dataset.num_parts),
          gamma(gamma),
	  decay(decay),
          batch_size(batch_size),
	  aggregate_batch_size(batch_size * dataset.num_parts),
	  num_model_updates(0),
          predicted_labels(std::make_unique<double[]>(
		           dataset.training_labels.num_classes * batch_size)) {
  if (svrg) {
    anchor_model = new double[model_size];
    sample_gradient = new double[model_size];
  }
}

void log_reg::multinomial_log_reg::train(const size_t num_epochs) {
    const size_t num_batches = get_num_batches();
    for (size_t epoch_num = 0; epoch_num < num_epochs; ++epoch_num) {
        for (size_t batch_num = 0; batch_num < num_batches; ++batch_num) {
	    compute_gradient(batch_num, model);
	    update_model();
        }
    }
}

// TO REMOVE
void log_reg::multinomial_log_reg::train_SVRG(const size_t num_epochs) {
  if (num_epochs % 3 == 1) {
    std::cout << "WARN: num_epochs should not be num_epochs % 3 == 1" << std::endl;
  }
  const size_t num_batches = get_num_batches();
  size_t left_epochs;
  for (size_t epoch_num = 0; epoch_num < std::ceil((double)num_epochs / (1 + num_inner_epochs)); ++epoch_num) {
    copy_model(model, anchor_model, model_size);
    compute_full_gradient(anchor_model);
    left_epochs = num_epochs - (1 + ((num_inner_epochs + 1) * epoch_num));
    if (left_epochs < num_inner_epochs) {
      num_inner_epochs = left_epochs;
    }
    for (size_t batch_num = 0; batch_num < num_batches * num_inner_epochs; ++batch_num) {
      compute_gradient(batch_num % num_batches, model);
      update_gradient(batch_num % num_batches);
      update_model();
    }
  }
}

void log_reg::multinomial_log_reg::copy_model(double* src, double* dst, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    dst[i] = src[i];
  }
}

double log_reg::multinomial_log_reg::training_error() {
    return compute_error(dataset.training_images, dataset.training_labels);
}

double log_reg::multinomial_log_reg::training_loss() {
    return compute_loss(dataset.training_images, dataset.training_labels);
}

double log_reg::multinomial_log_reg::get_loss_opt() const {
  std::ifstream loss_opt_file;
  std::string worker_dir = std::to_string(dataset.num_parts) + "workers";
  loss_opt_file.open(dataset.data_path + "/" + worker_dir + "/svrg_loss_opt.txt");
  std::string loss_opt_str;
  loss_opt_file >> loss_opt_str;
  double loss_opt = std::stod(loss_opt_str);
  return loss_opt;
}

double log_reg::multinomial_log_reg::test_error() {
    return compute_error(dataset.test_images, dataset.test_labels);
}

void log_reg::multinomial_log_reg::compute_gradient(size_t batch_num, double* given_model, double* gradient_ptr) {
    const utils::images_t& images = dataset.training_images;
    const utils::labels_t& labels = dataset.training_labels;
    batch_num += dataset.part_num * get_num_batches(); // convert from local batch_num to global batch_num
    
    utils::submatrix_multiply(CblasNoTrans, CblasNoTrans,
			      given_model, images.arr.get(), predicted_labels.get(),
			      0, 0, 0, batch_num * batch_size,
			      labels.num_classes, batch_size, images.num_pixels,
			      images.num_pixels, images.num_total_images,
			      1, 0);
    utils::softmax(predicted_labels.get(),
                   labels.num_classes, batch_size);
    cblas_dgeadd(CblasRowMajor, labels.num_classes, batch_size,
		 -1, labels.arr.get() + batch_num * batch_size, images.num_total_images,
		 1, predicted_labels.get(), batch_size);
    utils::submatrix_multiply(CblasNoTrans, CblasTrans,
                              predicted_labels.get(), images.arr.get(), gradient_ptr,
                              0, 0, 0, batch_num * batch_size,
                              labels.num_classes, images.num_pixels, batch_size, 
			      batch_size, images.num_total_images,
			      1/(double)batch_size, 0);
    cblas_daxpy(model_size, gamma, given_model, 1, gradient_ptr, 1);
}
// TO REMOVE
void log_reg::multinomial_log_reg::compute_gradient(size_t batch_num, double* given_model) {
    const utils::images_t& images = dataset.training_images;
    const utils::labels_t& labels = dataset.training_labels;
    batch_num += dataset.part_num * get_num_batches(); // convert from local batch_num to global batch_num
    
    double* gradient_ptr = NULL;
    if (given_model == anchor_model) {
      gradient_ptr = sample_gradient;
    } else {
      gradient_ptr = gradients[0][0];
    }
   
    utils::submatrix_multiply(CblasNoTrans, CblasNoTrans,
			      given_model, images.arr.get(), predicted_labels.get(),
			      0, 0, 0, batch_num * batch_size,
			      labels.num_classes, batch_size, images.num_pixels,
			      images.num_pixels, images.num_total_images,
			      1, 0);
    utils::softmax(predicted_labels.get(),
                   labels.num_classes, batch_size);
    cblas_dgeadd(CblasRowMajor, labels.num_classes, batch_size,
		 -1, labels.arr.get() + batch_num * batch_size, images.num_total_images,
		 1, predicted_labels.get(), batch_size);
    utils::submatrix_multiply(CblasNoTrans, CblasTrans,
                              predicted_labels.get(), images.arr.get(), gradient_ptr,
                              0, 0, 0, batch_num * batch_size,
                              labels.num_classes, images.num_pixels, batch_size, 
			      batch_size, images.num_total_images,
			      1/(double)batch_size, 0);
    cblas_daxpy(model_size, gamma, given_model, 1, gradient_ptr, 1);
}
// TO REMOVE
void log_reg::multinomial_log_reg::update_gradient(const size_t batch_num) {
  compute_gradient(batch_num, anchor_model); // compute sample_gradient using anchor_model
  cblas_daxpy(model_size, -1, sample_gradient, 1, gradients[0][0], 1);
  cblas_daxpy(model_size, 1, full_gradient, 1, gradients[0][0], 1);
}

void log_reg::multinomial_log_reg::compute_full_gradient(double* given_model) {
    const utils::images_t& images = dataset.training_images;
    const utils::labels_t& labels = dataset.training_labels;
    
    utils::submatrix_multiply(CblasNoTrans, CblasNoTrans,
			      given_model, images.arr.get(), full_predicted_labels.get(),
			      0, 0, 0, 0,
			      labels.num_classes, images.num_total_images, images.num_pixels,
			      images.num_pixels, images.num_total_images,
			      1, 0);
    utils::softmax(full_predicted_labels.get(),
                   labels.num_classes, images.num_total_images);
    cblas_dgeadd(CblasRowMajor, labels.num_classes, images.num_total_images,
		 -1, labels.arr.get(), images.num_total_images,
		 1, full_predicted_labels.get(), images.num_total_images);
    utils::submatrix_multiply(CblasNoTrans, CblasTrans,
                              full_predicted_labels.get(), images.arr.get(), full_gradient,
                              0, 0, 0, 0,
                              labels.num_classes, images.num_pixels, images.num_total_images, 
			      images.num_total_images, images.num_total_images,
			      1/(double)images.num_total_images, 0);
    cblas_daxpy(model_size, gamma, given_model, 1, full_gradient, 1);
}

void log_reg::multinomial_log_reg::update_model(uint ml_sst_row, uint ml_sst_col) {
    double decayed_alpha = decay_alpha();
    cblas_daxpy(model_size, -decayed_alpha,
		gradients[ml_sst_row-1][ml_sst_col], 1, model, 1);
    num_model_updates++;
}
// TO REMOVE
void log_reg::multinomial_log_reg::update_model(uint ml_sst_row) {
    double decayed_alpha = decay_alpha();
    cblas_daxpy(model_size, -decayed_alpha, gradients[ml_sst_row-1][0], 1, model, 1);
    num_model_updates++;
}
// TO REMOVE
void log_reg::multinomial_log_reg::update_model() {
    double decayed_alpha = decay_alpha();
    cblas_daxpy(model_size, -decayed_alpha, gradients[0][0], 1, model, 1);
    num_model_updates++;
}

double log_reg::multinomial_log_reg::decay_alpha() {
  double epoch = floor(num_model_updates / (get_num_batches() * dataset.num_parts));
  return alpha * pow(decay, epoch);
}

void log_reg::multinomial_log_reg::set_model_mem(double* model) {
    this->model = model;
}

void log_reg::multinomial_log_reg::initialize_model_mem_with_zero() {
    utils::zero_arr(this->model, model_size);
}

void log_reg::multinomial_log_reg::push_back_to_grads_vec(std::vector<double*> gradients) {
  this->gradients.push_back(gradients);
}

size_t log_reg::multinomial_log_reg::get_model_size() const {
    return model_size;
}

size_t log_reg::multinomial_log_reg::get_num_batches() const {
    return dataset.training_images.num_part_images / batch_size;
}

size_t log_reg::multinomial_log_reg::get_num_part_images() const {
    return dataset.training_images.num_part_images;
}

size_t log_reg::multinomial_log_reg::get_num_total_images() const {
    return dataset.training_images.num_total_images;
}

double* log_reg::multinomial_log_reg::get_model() const {
  return model;
}

double* log_reg::multinomial_log_reg::get_anchor_model() const {
  return anchor_model;
}

double* log_reg::multinomial_log_reg::get_full_gradient() const {
  return full_gradient;
}

double* log_reg::multinomial_log_reg::get_sample_gradient() const {
  return sample_gradient;
}

double log_reg::multinomial_log_reg::compute_error(const utils::images_t& images, const utils::labels_t& labels) {
    predicted_labels = std::make_unique<double[]>(
            labels.num_classes * images.num_total_images);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                labels.num_classes, images.num_total_images, images.num_pixels,
		1, model, images.num_pixels, images.arr.get(), images.num_total_images,
		0, predicted_labels.get(), images.num_total_images);
    utils::softmax(predicted_labels.get(),
                   labels.num_classes, images.num_total_images);
    size_t num_incorrect = 0;
    for(size_t i = 0; i < images.num_total_images; ++i) {
        if (cblas_idamax(labels.num_classes, predicted_labels.get() + i, images.num_total_images)
	    != cblas_idamax(labels.num_classes, labels.arr.get() + i, images.num_total_images)) {
	    num_incorrect++;
	}
    }
    return (double)num_incorrect/images.num_total_images;
}

double log_reg::multinomial_log_reg::compute_loss(const utils::images_t& images, const utils::labels_t& labels) {
    predicted_labels = std::make_unique<double[]>(
            labels.num_classes * images.num_total_images);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                labels.num_classes, images.num_total_images, images.num_pixels,
		1, model, images.num_pixels, images.arr.get(), images.num_total_images,
		0, predicted_labels.get(), images.num_total_images);
    utils::softmax(predicted_labels.get(),
                   labels.num_classes, images.num_total_images);

    double loss = 0.0;
    for (size_t i = 0; i < labels.num_classes * images.num_total_images; ++i) {
      if (*(labels.arr.get() + i) == 1) {
	loss += -std::log(*(predicted_labels.get() + i));
      }
    }
    loss /= images.num_total_images;

    for (size_t i = 0; i < labels.num_classes * images.num_pixels; ++i) {
      loss += (gamma / 2) * std::pow(model[i], 2);
    }
    return loss;
}

double log_reg::multinomial_log_reg::gradient_norm() {
  compute_full_gradient(model);
  double norm = 0.0;
  for (size_t i = 0; i < get_model_size(); ++i) {
    norm += std::pow(full_gradient[i], 2);
  }
  norm = sqrt(norm);
  return norm;
}

double log_reg::multinomial_log_reg::distance_to_optimum() {
  cnpy::NpyArray arr; 
  std::string worker_dir = std::to_string(dataset.num_parts) + "workers";
  arr = cnpy::npy_load(dataset.data_path + "/" + worker_dir + "/svrg_w_opt.npy");
  // Be careful that arr.data<double>() is std::shared_ptr, which means the allocated memory for data will be detroyed when out of scope.
  // That's why we didn't seperate above codes as a seperate function.
  double* model_opt = arr.data<double>();
  double norm = 0.0;
  for (size_t i = 0; i < get_model_size(); ++i) {
    norm += std::pow(model[i] - model_opt[i], 2);
  }
  norm = sqrt(norm);
  return norm;
}

void log_reg::multinomial_log_reg::save_npy_model() const {
  std::string worker_dir = std::to_string(dataset.num_parts) + "workers";
  if (svrg) {
    std::cout << "Saving a new svrg_w_opt.npy..." << std::endl;
    cnpy::npy_save(dataset.data_path + "/" + worker_dir + "/svrg_w_opt.npy",
		   model, {dataset.training_labels.num_classes, dataset.training_images.num_pixels}, "w");
  } else {
    std::cout << "Saving a new sgd_w_opt.npy..." << std::endl;
    cnpy::npy_save(dataset.data_path + "/" + worker_dir + "/sgd_w_opt.npy",
		   model, {dataset.training_labels.num_classes, dataset.training_images.num_pixels}, "w");
  }
}
