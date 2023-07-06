///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/ops/maxpoolingindice.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/core/maxpoolingindice.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <math.h>
#include <my_cblas.hpp>

namespace dlprim {
MaxPoolingIndiceMulConfig MaxPoolingIndiceMulConfig::from_json(json::value const &v)
{
    MaxPoolingIndiceMulConfig cfg;
    utils::get_1dNd_from_json(v,"kernel",cfg.kernel,true);
    utils::get_1dNd_from_json(v,"stride",cfg.stride);
    utils::get_1dNd_from_json(v,"dilate",cfg.dilate);
    utils::get_1dNd_from_json(v,"pad",cfg.pad);
    return cfg;
}


MaxPoolingIndice::MaxPoolingIndice(Context &ctx,MaxPoolingIndiceMulConfig config) :
    Operator(ctx),
    config_(config),
    dtype_(float_data)
{
    DLPRIM_CHECK(dtype_ == float_data);
}

MaxPoolingIndice::~MaxPoolingIndice()
{
}

void MaxPoolingIndice::setup(std::vector<TensorSpecs> const &in,std::vector<TensorSpecs> &out,std::vector<TensorSpecs> &indices,std::vector<TensorSpecs> &p,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    DLPRIM_CHECK(in[0].dtype() == dtype_);
    out.assign({in[0]});
    indices.assign({in[0]});
    p.clear();
    ws = 0;
}

void MaxPoolingIndice::reshape(std::vector<Shape> const &in,std::vector<Shape> &out,std::vector<Shape> &indices,size_t &ws)
{
    DLPRIM_CHECK(in.size()==1);
    out.assign({in[0]});
    indices.assign({in[0]});
    ws=0;
}

void MaxPoolingIndice::forward(std::vector<Tensor> &input,std::vector<Tensor> &output,std::vector<Tensor> &indices, std::vector<Tensor> &,Tensor &,ExecutionContext const &e)
{
    DLPRIM_CHECK(input.size()==1);
    DLPRIM_CHECK(output.size()==1); 
    DLPRIM_CHECK(indices.size()==1); 
    
    DLPRIM_CHECK(output[0].shape() == input[0].shape());
    DLPRIM_CHECK(indices[0].shape() == input[0].shape());
    
    DLPRIM_CHECK(input[0].dtype() == dtype_);
    DLPRIM_CHECK(output[0].dtype() == dtype_);

    core::forward(input[0],output[0],indices[0],config_.activation,e);
}


} // dlprim
