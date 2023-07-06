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

    
    class DMMForward{
    public:
        virtual ~DMMForward() {}
        virtual void enqueue(Tensor &x,Tensor &y,Tensor &indx,ExecutionContext const &e) = 0;
        static std::unique_ptr<DMMForward> create(Context &ctx,size_t M1,size_t M2,size_t M3,DataType dt,std::string method="max");
    };

} // core
} // dlprim
