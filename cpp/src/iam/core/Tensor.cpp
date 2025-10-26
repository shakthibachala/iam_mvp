#include "iam/core/Tensor.hpp"
#include <numeric>

namespace iam {
int Tensor::size() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
}
}
