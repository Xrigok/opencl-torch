///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/upsample_nearest2d.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/core/up2d.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <math.h>
#include <my_cblas.hpp>

namespace dlprim {
UpsampleNearest2dConfig UpsampleNearest2dConfig::from_json(json::value const &v)
{
    UpsampleNearest2dConfig cfg;
    cfg.s_h = v.get<int>("s_h",cfg.s_h);
    cfg.s_w = v.get<double>("s_w",cfg.s_w);
    return cfg;
}


UpsampleNearest2d::UpsampleNearest2d(Context &ctx,UpsampleNearest2dConfig config) :
    Operator(ctx),
    config_(config),
    dtype_(float_data)
{
    DLPRIM_CHECK(dtype_ == float_data);
}

UpsampleNearest2d::~UpsampleNearest2d()
{
}

void UpsampleNearest2d::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &p,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].dtype() == dtype_);
    out.assign({in[0]});
    p.clear();
    ws = 0;
}

void UpsampleNearest2d::reshape(std::vector<Shape> const &in,std::vector<Shape> &out,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    out.assign({in[0]});
    ws=0;
}

void UpsampleNearest2d::forward(std::vector<Tensor> &input,std::vector<Tensor> &output, std::vector<Tensor> &,Tensor &,ExecutionContext const &e)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    
    DLPRIM_CHECK(output[0].shape() == input[0].shape());
    
    DLPRIM_CHECK(input[0].dtype() == dtype_);
    DLPRIM_CHECK(output[0].dtype() == dtype_);

    core::forward(input[0],output[0],e);
}


} // dlprim
