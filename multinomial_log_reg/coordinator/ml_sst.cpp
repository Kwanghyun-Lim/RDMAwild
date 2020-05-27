#include "ml_sst.hpp"
#include "utils/utils.hpp"

sst::MLSST::MLSST(const std::vector<uint32_t>& members,
                  const uint32_t my_id, const size_t num_params)
        : sst::SST<MLSST>(this, SSTParams{members, my_id}),
          model_or_gradient(num_params),
	  last_round(16),
          model_or_gradient1(num_params),
	  last_round1(16),
          model_or_gradient2(num_params),
	  last_round2(16)
{
    SSTInit(model_or_gradient, round, last_round,
	    model_or_gradient1, round1, last_round1,
	    model_or_gradient2, round2, last_round2);
    initialize();
}

void sst::MLSST::initialize() {
    uint32_t num_nodes = get_num_rows();
    for(uint i = 0; i < num_nodes; ++i) {
      utils::zero_arr((double*)std::addressof(
					      model_or_gradient[i][0]),
		      model_or_gradient.size());
      utils::zero_arr((double*)std::addressof(
					      model_or_gradient1[i][0]),
		      model_or_gradient1.size());
      utils::zero_arr((double*)std::addressof(
					      model_or_gradient2[i][0]),
		      model_or_gradient2.size());
    }
    
    for(uint i = 0; i < num_nodes; ++i) {
        round[i] = 0;
        round1[i] = 0;
        round2[i] = 0;
    }
    
    for(uint i = 0; i < num_nodes; ++i) {
      for(uint j = 0; j < last_round.size(); ++j) {
        last_round[i][j] = 0;
        last_round1[i][j] = 0;
        last_round2[i][j] = 0;
      }
    }
    
    put_with_completion();  // end of initialization
    sync_with_members();
}
