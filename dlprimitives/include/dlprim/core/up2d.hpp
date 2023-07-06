///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/tensor.hpp>
#include <dlprim/context.hpp>
namespace dlprim {
namespace core {

    
    class UpsampleNearest2dFwd{
    public:
        virtual ~UpsampleNearest2dFwd() {}
        virtual void enqueue(Tensor &in,Tensor &out,ExecutionContext const &e) = 0;
        static std::unique_ptr<UpsampleNearest2dFwd> create(Context &ctx,Shape const &ishape,Shape const &oshape,double s_h,double s_w,DataType dt);
    };

} // core
} // dlprim
