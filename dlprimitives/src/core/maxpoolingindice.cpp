///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/maxpoolingindice.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include<iostream>

namespace dlprim {
namespace core {
    class MaxPooling2DIndicesFWDImpl : public MaxPooling2DIndicesForward {
    public:
        size_t workspace() { return 0; }
        MaxPooling2DIndicesFWDImpl(Context &ctx,Shape const &x_shape,Shape const &y_shape,int k[2],int p[2],int s[2],int d[2],DataType dt)
        {
            DLPRIM_CHECK(dt == float_data);
            wg_size_=16;
            NC=y_shape[0];
            if(y_shape.size()==4){
                NC*=y_shape[1];
            }
            OH=y_shape[y_shape.size()-2];
            OW=y_shape[y_shape.size()-1];

            cl::Program const &prog = gpu::Cache::instance().get_program(ctx,"maxpoolingindice",
                                                        "kx",k[0],
                                                        "ky",k[1],
                                                        "sx",s[0],
                                                        "sy",s[1],
                                                        "px",p[0],
                                                        "py",p[1],
                                                        "dx",d[0],
                                                        "dy",d[1],
                                                        "IH",int(x_shape[x_shape.size()-2]),
                                                        "IW",int(x_shape[x_shape.size()-1]),
                                                        "OH",int(y_shape[y_shape.size()-2]),
                                                        "OW",int(y_shape[y_shape.size()-1]));
            kernel_ = cl::Kernel(prog,"maxpoolingindice_forward");
        }
        void enqueue(Tensor &in,Tensor &out,Tensor &indices,ExecutionContext const &ctx){
            int p=0;
            in.set_arg(kernel_,p);
            out.set_arg(kernel_,p);
            indices.set_arg(kernel_,p);

            
            cl::NDRange wg={1,wg_size_,wg_size_};
            cl::NDRange gr={NC,gpu::round_up(OH,wg_size_),gpu::round_up(OW,wg_size_)};
            ctx.queue().enqueueNDRangeKernel(kernel_,cl::NullRange,gr,wg,ctx.events(),ctx.event("maxpoolingindice_forward"));
        }

    private:
        int wg_size_;
        cl::Kernel kernel_;
        int NC,OH,OW;
    };

    std::unique_ptr<MaxPooling2DIndicesForward> MaxPooling2DIndicesForward::create(Context &ctx,Shape const &x_shape,Shape const &y_shape,int k[2],int p[2],int s[2],int d[2],DataType dt)
    {
        std::unique_ptr<MaxPooling2DIndicesForward> r(new MaxPooling2DIndicesFWDImpl(ctx,x_shape,y_shape,k,p,s,d,dt));
        return r;
    }

} // core
} // dlprim
