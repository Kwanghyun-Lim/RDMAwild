#pragma once

#include <vector>

#include "sst.h"

namespace sst {
class MLSST : public SST<MLSST> {
public:
    MLSST(const std::vector<uint32_t>& members, const uint32_t my_id,
          const size_t num_params);

    SSTFieldVector<double> model_or_gradient;
    SSTField<uint64_t> round;
    SSTFieldVector<uint64_t> last_round;
  
    SSTFieldVector<double> model_or_gradient1;
    SSTField<uint64_t> round1;
    SSTFieldVector<uint64_t> last_round1;
  
    SSTFieldVector<double> model_or_gradient2;
    SSTField<uint64_t> round2;
    SSTFieldVector<uint64_t> last_round2;

private:
    void initialize();
};
}  // namespace sst
