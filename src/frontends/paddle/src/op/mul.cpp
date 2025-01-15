// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
OutputVector reshape_by_axis(const Output<Node>& node, int aixs, int rank, bool return_left_dims) {
    if (aixs < 0) {
        aixs += rank;
    }
    auto shape_of_node = std::make_shared<default_opset::ShapeOf>(node);
    auto axis1_begin = default_opset::Constant::create(element::i64, {1}, {0});
    auto axis1_end = default_opset::Constant::create(element::i64, {1}, {aixs});
    auto front_slice_node = std::make_shared<default_opset::StridedSlice>(shape_of_node,
                                                                          axis1_begin,
                                                                          axis1_end,
                                                                          std::vector<int64_t>{0},
                                                                          std::vector<int64_t>{0});
    auto front_node = std::make_shared<default_opset::ReduceProd>(
        front_slice_node,
        default_opset::Constant::create(ov::element::i32, {1}, std::vector<int64_t>{0}),
        true);

    auto axis2_begin = default_opset::Constant::create(element::i64, {1}, {aixs});
    auto axis2_end = default_opset::Constant::create(element::i64, {1}, {rank});
    auto end_slice_node = std::make_shared<default_opset::StridedSlice>(shape_of_node,
                                                                        axis2_begin,
                                                                        axis2_end,
                                                                        std::vector<int64_t>{0},
                                                                        std::vector<int64_t>{0});
    auto end_node = std::make_shared<default_opset::ReduceProd>(
        end_slice_node,
        default_opset::Constant::create(ov::element::i32, {1}, std::vector<int64_t>{0}),
        true);

    OutputVector axes{front_node, end_node};
    auto reshape_node = std::make_shared<default_opset::Concat>(axes, 0);
    return OutputVector{std::make_shared<default_opset::Reshape>(node, reshape_node, false),
                        return_left_dims ? front_slice_node : end_slice_node};
}
NamedOutputs mul(const NodeContext& node) {
    auto x = node.get_input("X");
    auto y = node.get_input("Y");
    auto x_rank = static_cast<int>(x.get_partial_shape().size());
    auto y_rank = static_cast<int>(y.get_partial_shape().size());
    auto x_num_col_dims = node.get_attribute<int>("x_num_col_dims", 1);
    auto y_num_col_dims = node.get_attribute<int>("y_num_col_dims", 1);
    auto reshape_x_list = reshape_by_axis(x, x_num_col_dims, x_rank, true);
    auto reshape_y_list = reshape_by_axis(y, y_num_col_dims, y_rank, false);
    std::shared_ptr<Node> res;
    res = std::make_shared<default_opset::MatMul>(reshape_x_list[0], reshape_y_list[0]);
    if (x_rank != 2 || y_rank != 2) {
        auto reshape_node =
            std::make_shared<default_opset::Concat>(OutputVector{reshape_x_list[1], reshape_y_list[1]}, 0);
        res = std::make_shared<default_opset::Reshape>(res, reshape_node, false);
    }
    return node.default_single_output_mapping({res}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
