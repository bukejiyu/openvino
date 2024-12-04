// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset15.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs scatter(const NodeContext& node) {
    auto x = node.get_input("X");
    auto ids = node.get_input("Ids");
    auto updates = node.get_input("Updates");
    bool overwrite = node.get_attribute<bool>("overwrite");
    // 只有1维或者0维
    //  auto ids_shape=std::make_shared<default_opset::ShapeOf>(ids);
    ov::NodeVector node_vec;
    if (ids.get_shape().size() == 0) {
        ids = std::make_shared<default_opset::Unsqueeze>(ids,
                                                         default_opset::Constant::create(ov::element::i64, {1}, {0}));                                              
    }
    auto cast = std::make_shared<default_opset::Convert>(ids, element::i64);
    node_vec.push_back(cast);   
    node_vec.push_back(default_opset::Constant::create(ov::element::i64, {1}, {0}));
    auto shape_node = std::make_shared<default_opset::Concat>(node_vec, 0);
    // std::vector<int32_t> vec = {static_cast<int32_t>(ids.get_shape()[0]), 1};
    // auto shape_node = default_opset::Constant::create(ov::element::i32, ov::Shape{2}, vec);
    auto new_ids = std::make_shared<default_opset::Reshape>(ids, shape_node, true);
    // const auto dim_node = default_opset::Constant::create(element::i32, {1}, {0});
    // auto update_shape = std::make_shared<default_opset::ShapeOf>(updates);
    // auto update_shape_1 = std::make_shared<default_opset::ConvertLike>(update_shape, dim_node);
    // auto axis_rank = std::make_shared<default_opset::Add>(dim_node, update_shape_1);
    // auto new_axis = std::make_shared<default_opset::Mod>(axis_rank, update_shape_1);

    // auto const_1_vec = default_opset::Constant::create(element::i32, Shape{1}, {1});
    // auto dim_vec = std::make_shared<default_opset::Reshape>(new_axis, const_1_vec, false);
    // auto broadcasted_index = std::make_shared<default_opset::Broadcast>(index, update_shape, dim_vec);
    if (overwrite) {
        // 直接更新 并且按顺序 同torch dim=0
        // return node.default_single_output_mapping(
        //     {std::make_shared<ov::opset12::ScatterElementsUpdate>(x, broadcasted_index, updates, dim_node)},
        //     {"Out"});
        return node.default_single_output_mapping({std::make_shared<ov::opset15::ScatterNDUpdate>(x, new_ids, updates)},
                                                  {"Out"});
    } else {
        // 变成 sactter_nd tmp=0 然后 叠加更新
        auto x_dtype = x.get_element_type();
        // const std::vector<element::Type> supported_type =
        //     {element::boolean, element::i16, element::i32, element::i64, element::f16, element::f32, element::f64};
        // const bool valid_type =
        //     std::any_of(supported_type.begin(), supported_type.end(), [x_dtype](const element::Type& type) {
        //         return x_dtype == type;
        //     });
        const auto value_node = default_opset::Constant::create(x_dtype, {1}, {0});
        const auto shape_node = std::make_shared<default_opset::ShapeOf>(x);
        const auto zero_node = std::make_shared<default_opset::Broadcast>(value_node, shape_node);
        return node.default_single_output_mapping(
            {std::make_shared<ov::opset15::ScatterNDUpdate>(zero_node,
                                                            new_ids,
                                                            updates,
                                                            ov::opset15::ScatterNDUpdate::Reduction::SUM)},
            {"Out"});
    }
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
