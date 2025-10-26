#pragma once
#include <vector>

namespace iam {
class Tensor {
public:
    Tensor(std::vector<int> shape) : shape_(shape) {}
    std::vector<int> shape() const { return shape_; }
    int size() const;
private:
    std::vector<int> shape_;
};
}
