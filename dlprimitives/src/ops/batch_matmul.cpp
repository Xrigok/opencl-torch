///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/batch_matmul.hpp>
#include <dlprim/json.hpp>
#include <dlprim/core/common.hpp>
#include <dlprim/core/bmm.hpp>
#include <dlprim/ops/initialization.hpp>
#include <cmath>
#include <my_cblas.hpp>
#include <iostream>

namespace dlprim {
        BatchMatMul::BatchMatMul(Context &ctx,BatchMatMulConfig const &config,DataType dt) :
            Operator(ctx),
            config_(config),
            dtype_(dt)
        {
        }

        BatchMatMulConfig BatchMatMulConfig::from_json(json::value const &v) 
        {
            BatchMatMulConfig cfg;
            cfg.M1 = v.get<int>("M1",cfg.M1);
            cfg.M2 = v.get<int>("M2",cfg.M1);
            cfg.K = v.get<int>("K",cfg.M1);
            return cfg;
        }

        BatchMatMul::~BatchMatMul() {}
        void BatchMatMul::setup(std::vector<TensorSpecs> const &in,
                                std::vector<TensorSpecs> const &weight,
                                std::vector<TensorSpecs> &out,
                                std::vector<TensorSpecs> &parameters)
        {
            DLPRIM_CHECK(in.size()==1);
            DLPRIM_CHECK(in[0].shape().size() >= 2);

            DLPRIM_CHECK(weight.size()==1);
            DLPRIM_CHECK(weight[0].shape().size() >= 2);
            
            out = in;
            parameters.clear();
            Shape in_shape = in.shape();
            Shape w_shape = weight.shape();

            bmm_gpu_ = std::move(core::BMMForward::create(ctx_,in[0].shape(),weight[0].shape(),dtype_));
            Shape output_shape = get_output_shape(in_shape,w_shape);
        }
        

		
        
        void BatchMatMul::forward(std::vector<Tensor> &input,
                                  std::vector<Tensor> &weight,
                                  std::vector<Tensor> &output,
                                  ExecutionContext const &e)
        {
            ExecutionContext elast;
            elast = e;

            bmm_gpu_->enqueue_forward(
                        input[0],weight[0],output[0],
                        e.generate_series_context(2,3));
        }

        Shape BatchMatMul::get_output_shape(Shape const &in,Shape const &weight)
        {
            DLPRIM_CHECK(in.size() == 3);
            DLPRIM_CHECK(in.size() == 3);
            int batch = in[0];
            return Shape(batch,config_.channels_out,in.shape()[1],weight.shape()[1]);
        }



} // namesapce
