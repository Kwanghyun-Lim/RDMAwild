#include "numpy_reader.hpp"

#include <cstdint>
#include <exception>


numpy::numpy_dataset::numpy_dataset(const std::string& numpy_dir, const uint32_t num_parts, const uint32_t part_num)
  : reader(numpy_dir), dataset(numpy_dir, num_parts, part_num) {
    // std::cout << "[numpy_dataset] numpy_dir " << numpy_dir << std::endl;
    training_images = reader.read_numpy_image_file(training_images_filename, num_parts, part_num);
    training_labels = reader.read_numpy_label_file(training_labels_filename, num_parts, part_num);
    test_images = reader.read_numpy_image_file(test_images_filename, num_parts, part_num);
    test_labels = reader.read_numpy_label_file(test_labels_filename, num_parts, part_num);
}

numpy::numpy_reader::numpy_reader(const std::string& numpy_dir)
        : numpy_dir(numpy_dir) {
}

utils::images_t numpy::numpy_reader::read_numpy_image_file(const std::string& filename, const uint32_t num_parts, const uint32_t part_num) const {
    cnpy::NpyArray npy = cnpy::npy_load(utils::fullpath(numpy_dir, filename));
    utils::images_t images;
    const uint32_t num_total_images = npy.shape[1];
    images.num_total_images = num_total_images;
    images.num_part_images = num_total_images/num_parts;
    images.num_pixels = npy.shape[0];
    // std::cout << filename << ": num_total_images " << images.num_total_images
    // 	      << " num_part_images " << images.num_part_images
    // 	      << " num_pixels " << images.num_pixels << std::endl;
    double* npy_arr = npy.data<double>();
    images.arr = std::make_unique<double[]>(
			    images.num_total_images * images.num_pixels);
					    
    for (size_t i = 0; i < images.num_total_images * images.num_pixels; i++) {
      images.arr[i] = npy_arr[i];
    }
    
    return images;
}

utils::labels_t numpy::numpy_reader::read_numpy_label_file(const std::string& filename, const uint32_t num_parts, const uint32_t part_num) const {
    cnpy::NpyArray npy = cnpy::npy_load(utils::fullpath(numpy_dir, filename));
    utils::labels_t labels;
    const uint32_t num_total_labels = npy.shape[1];
    labels.num_total_labels = num_total_labels;
    labels.num_part_labels = num_total_labels/num_parts;
    labels.num_classes = npy.shape[0];
    // std::cout << filename << ": num_total_labels " << labels.num_total_labels
    //           << " num_part_labels " <<  labels.num_part_labels
    // 	      << " num_classes " << labels.num_classes << std::endl;
    double* npy_arr = npy.data<double>();
    labels.arr = std::make_unique<double[]>(
			    labels.num_total_labels * labels.num_classes);
					    
    for (size_t i = 0; i < labels.num_total_labels * labels.num_classes; i++) {
      labels.arr[i] = npy_arr[i];
    }
    
    return labels;
}

