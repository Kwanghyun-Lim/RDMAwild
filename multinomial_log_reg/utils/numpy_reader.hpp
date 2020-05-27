#pragma once

#include <memory>
#include <string>

#include "utils.hpp"
#include "cnpy.hpp"

namespace numpy {
class numpy_reader {
public:
    numpy_reader(const std::string& numpy_dir);

    utils::images_t read_numpy_image_file(const std::string& filename, const uint32_t num_parts, const uint32_t part_num) const;
    utils::labels_t read_numpy_label_file(const std::string& filename, const uint32_t num_parts, const uint32_t part_num) const;

private:
    const std::string numpy_dir;
};

class numpy_dataset : public utils::dataset {
public:
    numpy_dataset(const std::string& numpy_dir, const uint32_t num_parts, const uint32_t part_num);

private:
    const numpy_reader reader;

    const std::string training_images_filename = "Xs_tr.npy";
    const std::string test_images_filename = "Xs_te.npy";
    const std::string training_labels_filename = "Ys_tr.npy";
    const std::string test_labels_filename = "Ys_te.npy";
};
} // namespace numpy
