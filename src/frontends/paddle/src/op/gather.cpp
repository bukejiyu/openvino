// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs gather(const NodeContext& node) {
    const auto data_node = node.get_input("X");
    auto index_node = node.get_input("Index");
    Output<Node> axis_node;
    // std::cout << "[gather] data_node:" << static_cast<int>(data_node.get_partial_shape().size()) << "\n";
    // std::cout << "[gather] index_node:" << static_cast<int>(index_node.get_partial_shape().size()) << "\n";
    if (node.has_input("Axis")) {
        axis_node = node.get_input("Axis");
    } else {
        const auto axis_value = node.get_attribute<int>("axis", 0);
        axis_node = default_opset::Constant::create(element::i32, Shape{}, {axis_value});
    }
    auto index_dims = index_node.get_partial_shape().size();
    if (index_dims == 2) {
        index_node = std::make_shared<default_opset::Squeeze>(
            index_node,
            default_opset::Constant::create(ov::element::i32, {1}, std::vector<int64_t>{1}));
    }

    return node.default_single_output_mapping(
        {std::make_shared<default_opset::Gather>(data_node, index_node, axis_node)},
        {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
