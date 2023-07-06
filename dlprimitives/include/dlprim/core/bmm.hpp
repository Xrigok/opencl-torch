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

    
    class BMMForward{
    public:
        virtual ~BMMForward() {}
        virtual void enqueue(Tensor &x,Tensor &w,Tensor &y,ExecutionContext const &e) = 0;
        static std::unique_ptr<BMMForward> create(Context &ctx,Shape const &s1,Shape const &s2,DataType dt);
    };

} // core
} // dlprim
