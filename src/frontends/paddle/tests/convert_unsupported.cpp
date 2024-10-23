// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/manager.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset6.hpp"
#include "paddle_utils.hpp"
#include "utils.hpp"

using namespace ov::frontend;

TEST(FrontEndConvertModelTest, test_unsupported_op) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    OV_ASSERT_NO_THROW(frontEnd = fem.load_by_framework(PADDLE_FE));
    ASSERT_NE(frontEnd, nullptr);
    // auto model_filename = FrontEndTestUtils::make_model_path(std::string(TEST_PADDLE_MODELS_DIRNAME) +
    //                                                          std::string("relu_unsupported/relu_unsupported.pdmodel"));
    auto model_filename=FrontEndTestUtils::make_model_path("/workspace/cpu_anlysis/pnc_test/paddle_cpu/jl_vru/left_turn_vru_multipath_plus_model.pdmodel");
    OV_ASSERT_NO_THROW(inputModel = frontEnd->load(model_filename));
    ASSERT_NE(inputModel, nullptr);
    std::shared_ptr<ov::Model> model;
    model = frontEnd->convert_partially(inputModel);
    for (auto& node : model->get_ordered_ops()) {
        std::cout<<"node->description():"<<node->description()<<"\n"<<std::endl;
        if (node->description() == "FrameworkNode") {
            std::cout<<"不支持的op node->get_friendly_name():"<<node->get_friendly_name()<<"\n"<<std::endl;
        }
    }

    // std::shared_ptr<ov::Model> model;
    // ASSERT_THROW(model = frontEnd->convert(inputModel), OpConversionFailure);
    // ASSERT_EQ(model, nullptr);
    // OV_ASSERT_NO_THROW(model = frontEnd->decode(inputModel));
    // ASSERT_THROW(frontEnd->convert(model), OpConversionFailure);
    // OV_ASSERT_NO_THROW(model = frontEnd->convert_partially(inputModel));
    // ASSERT_THROW(frontEnd->convert(model), OpConversionFailure);

    // for (auto& node : model->get_ordered_ops()) {
    //     if (node->get_friendly_name() == "rxyz_0.tmp_0") {
    //         model->replace_node(node, std::make_shared<ov::opset6::Relu>(node->input(0).get_source_output()));
    //     }
    // }
    // OV_ASSERT_NO_THROW(frontEnd->convert(model));
}
