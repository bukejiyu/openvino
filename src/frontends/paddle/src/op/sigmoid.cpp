// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs sigmoid(const NodeContext& node) {
    auto data = node.get_input("X");
    std::cout << "[sigmoid] x:" << static_cast<int>(data.get_partial_shape().size());
    return node.default_single_output_mapping({std::make_shared<ov::opset6::Sigmoid>(data)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
