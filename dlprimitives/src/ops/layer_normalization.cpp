///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/layer_normalization.hpp>
#include <dlprim/json.hpp>
#include <dlprim/core/common.hpp>
#include <dlprim/core/bn.hpp>
#include <dlprim/ops/initialization.hpp>
#include <cmath>
#include <my_cblas.hpp>
#include <iostream>

namespace dlprim {
        LayerNorm::LayerNorm(Context &ctx,LayerNormConfig const &config,DataType dt) :
            Operator(ctx),
            config_(config),
            dtype_(dt)
        {
        }

        LayerNormConfig LayerNormConfig::from_json(json::value const &v) 
        {
            LayerNormConfig cfg;
            cfg.features = v.get<int>("features",cfg.features);
            cfg.eps = v.get<float>("eps",cfg.eps);
            cfg.affine = v.get<bool>("affine",cfg.affine);
            cfg.use_global_stats = v.get<bool>("use_global_stats",cfg.use_global_stats);
            return cfg;
        }

        LayerNorm::~LayerNorm() {}
        void LayerNorm::setup(std::vector<TensorSpecs> const &in,
                                std::vector<TensorSpecs> &out,
                                std::vector<TensorSpecs> &parameters)
        {
            DLPRIM_CHECK(in.size()==1);
            DLPRIM_CHECK(in[0].shape().size() >= 2);
            
            out = in;
            parameters.clear();

            ln_gpu_ = std::move(core::LayerNormFwdBwd::create(ctx_,in[0].shape(),dtype_));
            setup_shape_ = in[0].shape();
        }
        void LayerNorm::mode(CalculationsMode m)
        {
            Operator::mode(m);
        }
        
        void LayerNorm::initialize_params(std::vector<Tensor> &parameters,ExecutionContext const &e)
        {
            if(config_.affine) {
                set_to_constant(parameters.at(0),1.0,e);
                set_to_zero(parameters.at(1),e);
            }
        }
        
		
        
        void LayerNorm::forward(std::vector<Tensor> &input,
                                  std::vector<Tensor> &output,
                                  std::vector<Tensor> &parameters,
                                  Tensor &ws,
                                  ExecutionContext const &e)
        {
            ExecutionContext elast;
            elast = e;
            if(config_.affine) {
                ln_gpu_->enqueue_forward_affine(
                        input[0],output[0],
                        parameters.at(0),parameters.at(1),
                        config_.eps,
                        ws,e.generate_series_context(2,3));
            }
            else {
                ln_gpu_->enqueue_forward_direct(
                        input[0],output[0],
                        config_.eps,
                        ws,e.generate_series_context(2,3));
            }
        }




} // namesapce
