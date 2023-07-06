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

    class MaxPooling2DIndicesForward {
    public:
        virtual ~MaxPooling2DIndicesForward() {}
        // get workspace size
        virtual size_t workspace() = 0;

        ///
        /// when used with kernel based pooling (not global)
        /// X and Y dimensions should match at batch and channels and for H/W  the dimention for Y should be Y_dim = op((X_dim + 2 * pad[dim] - kernel[dim]) / stride[dim]) + 1
        /// where op is either ceil or floor
        ///
        virtual void enqueue(Tensor &X,Tensor &Y,Tensor &inds,ExecutionContext const &e) = 0;

        ///
        /// Create max pooling for kernel, pad, stride
        ///
        static std::unique_ptr<MaxPooling2DIndicesForward> create(
                            Context &ctx,Shape const &x_shape,Shape const &y_shape,
                            int kernel[2],int pad[2],int stride[2],int dilation[2],
                            DataType dt=float_data);

    };
} // core
} //dlprim
