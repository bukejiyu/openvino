// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

NamedOutputs data_norm(const NodeContext& node) {
    auto x = node.get_input("X");
    auto bs = node.get_input("BatchSize");
    auto batch_square_sum = node.get_input("BatchSquareSum");
    auto batch_sum = node.get_input("BatchSum");
    // 1.mean = batch_sum / bs
    auto mean_out = std::make_shared<default_opset::Divide>(batch_sum, bs);
    // 2.scale = bs / batch_square_sum
    Output<Node> scale_out = std::make_shared<default_opset::Divide>(bs, batch_square_sum);
    scale_out = std::make_shared<default_opset::Sqrt>(scale_out);
    // 3.out = x-mean
    auto res_node = std::make_shared<default_opset::Subtract>(x, mean_out);
    NamedOutputs named_outputs;
    named_outputs["Y"] = {res_node};
    named_outputs["Means"] = {mean_out};
    named_outputs["Scales"] = {scale_out};
    return named_outputs;
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
