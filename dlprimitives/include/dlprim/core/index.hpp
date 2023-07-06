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

    
    class IndexSelectForward{
    public:
        virtual ~IndexSelectForward() {}
        virtual void enqueue(Tensor &input,Tensor &XX,Tensor &DS,Tensor &out,ExecutionContext const &e) = 0;
        static std::unique_ptr<IndexSelectForward> create(Context &ctx,long M2,long M3,long D1,
                                                            long D2,long OD1,long L,size_t out_size,DataType dt);
    };
    
    class IndexPutForward{
    public:
        virtual ~IndexPutForward() {}
        virtual void enqueue(Tensor &input,Tensor &XX,Tensor &DS,Tensor &value,ExecutionContext const &e) = 0;
        static std::unique_ptr<IndexPutForward> create(Context &ctx,long D1,long DA,long DC,long M2,long L,DataType dt);
    };

} // core
} // dlprim
