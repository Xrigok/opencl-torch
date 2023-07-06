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
#include <iostream>
namespace dlprim {
namespace core {
    ///
    /// Performs batch normalization computations over channel #1 (when #0 is batch)
    ///
    /// Pseudo code parameters:
    ///
    ///   \code 
    ///     // Layer Data
    ///     Tensor running_mean,running_var,gamma,beta;
    ///     // Temorary Data kept between FW and BW
    ///     Tensor mean,var;
    ///     // Workspace
    ///     Tensor ws;
    ///   \endcode
    ///
    ///  Actual pseudo code calcultions
    ///   Affine, Train
    ///   \code 
    ///     // Forward Pass
    ///     enqueue_calculate_batch_stats(x,mean,var,ws)
    ///     enqueue_update_running_stats(0.1,0.9,mean,running_mean,
    ///                                  0.1 * m/(m-1),0.9,var,running_var,ws);
    ///     enqueue_forward_affine(x,y, gamma,beta, mean, var,ws);
    ///
    ///     // Backward pass
    ///     enqueue_backward_affine(true,x,dy,mean,var,gamma,&dx,&dgamma,&dbeta,ws);
    ///  \endcode
    ///
    ///   Affine, Test (fixed batch)
    ///   \code 
    ///     // Forward Pass
    ///     enqueue_forward_affine(x,y, gamma,beta, running_mean, running_var,ws);
    ///
    ///     // Backward pass
    ///     enqueue_backward_affine(false,x,dy,running_mean,runnig_var,gamma,&dx,&dgamma,&dbeta,ws);
    ///  \endcode
    ///
    ///   Without affine, Train
    ///   \code 
    ///     // Forward Pass
    ///     enqueue_calculate_batch_stats(x,mean,var,ws)
    ///     enqueue_update_running_stats(0.1,0.9,mean,running_mean,
    ///                                  0.1 * m/(m-1),0.9,var,running_var,ws);
    ///     enqueue_forward_direct(x,y, mean, var,ws);
    ///
    ///     // Backward pass
    ///     enqueue_backward_direct(true,x,dy,mean,var,dx,ws);
    ///  \endcode
    ///
    ///   without affine, Test (fixed batch)
    ///   \code 
    ///     // Forward Pass
    ///     enqueue_forward_direct(x,y, running_mean, running_var,ws);
    ///
    ///     // Backward pass
    ///     enqueue_backward_direct(false,x,dy,running_mean,runnig_var,dx,ws);
    ///  \endcode
    ////

    class LayerNormFwdBwd {
    public:
        virtual ~LayerNormFwdBwd() {}
        
        ///
        /// Workspace size needed for intermediate results of computations
        ///

        
        ///
        /// Compute batch mean and variance for input x
        ///
        /// Note \a mean and \a var shoudl have Shape(features) where features is x.shape()[1]
        ///
        

        ///
        /// Peform forward computation as y = (x-mean) / sqrt(var + eps)
        ///
        /// Note mean/var can be taken from batch or from global running stats as per user request
        ///
        virtual void enqueue_forward_direct(Tensor &x,Tensor &y,float eps,
                                            ExecutionContext const &e) = 0;
        ///
        /// Peform forward computation as y = (x-mean) / sqrt(var + eps) * gamma + beta 
        ///
        /// Notes:
        /// - mean/var can be taken from batch or from global running stats as per user request
        /// - mean/var and gamma/beta are converted to single y=ax+b and than computation is done in a single step
        ///
        virtual void enqueue_forward_affine(Tensor &x,Tensor &y,
                                            Tensor &gamma,Tensor &beta,
                                            float eps,
                                            ExecutionContext const &e) = 0;


        static std::unique_ptr<LayerNormFwdBwd> create(Context &ctx,Shape const &s,long const &ns,DataType dt=float_data); //
        
    };

} // core
} // dlprim
