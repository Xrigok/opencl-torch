///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/common.hpp>
#include <dlprim/core/flip.hpp>
#include <dlprim/gpu/program_cache.hpp>

#include <iostream>
#include <sstream>

namespace dlprim {
namespace core {
    void flip(Tensor &x,Tensor &y,Tensor &DL,Tensor &Dims,ExecutionContext const &e){
    
        DLPRIM_CHECK(x.shape().total_size());
        DLPRIM_CHECK(y.shape().total_size());
        
        Context ctx(e);

        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"flip",
                                                                     "dtype", data_type_to_opencl_type(x.dtype()),
                                                                     "L1", DL.shape()[0],
                                                                     "L2", Dims.shape()[0]);
        cl::Kernel k(prog,"flip_exec");
        int p=0;
        x.set_arg(k,p);
        y.set_arg(k,p);
        DL.set_arg(k,p);
        Dims.set_arg(k,p);
        e.queue().enqueueNDRangeKernel(k,cl::NullRange,x.shape().total_size(),cl::NullRange,e.events(),e.event("flip"));
    }
} // core
} // dlprim

