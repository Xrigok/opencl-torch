///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/up2d.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <iostream>

namespace dlprim {
namespace core {
    class UpsampleNearest2dImpl : public UpsampleNearest2dFwd {
    public:
        virtual ~UpsampleNearest2dImpl() {}

        UpsampleNearest2dImpl(Context &ctx,Shape const &ishape,Shape const &oshape,double s_h,double s_w,DataType dtype)
        {
            dt_ = dtype;
            DLPRIM_CHECK(dtype == float_data);
            scale_h=s_h;
            scale_w=s_w;
            
            IH=int(ishape[2]);
            IW=int(ishape[3]);
            OH=int(oshape[2]);
            OW=int(oshape[3]);

            local_size={1,16,16};

            int gs1=(OH+16-1)/16*16;
            int gs2=(OW+16-1)/16*16;
            global_size={int(ishape[0])*int(ishape[1]),gs1,gs2};

            cl::Program const &utils = gpu::Cache::instance().get_program(ctx,"up2d","OH",OH,
                                                                                    "OW",OW,
                                                                                    "IH",IH,
                                                                                    "IW",IW);
            
            forward_ = cl::Kernel(utils,"up2d_forward");
        }

        virtual void enqueue(Tensor &in,Tensor &out,ExecutionContext const &e)
        {

            int p = 0;
            in.set_arg(forward_,p);
            out.set_arg(forward_,p);

            forward_.setArg(p++,scale_h);
            forward_.setArg(p++,scale_w);
            e.queue().enqueueNDRangeKernel(forward_,cl::NullRange,global_size,local_size,e.events(),e.event("upsample_nearest2d"));

        }
    
    private:
        double scale_h,scale_w;
        int IH,IW,OH,OW;
        cl::NDRange global_size;
        cl::NDRange local_size;

        DataType dt_;
        cl::Kernel forward_;
    };

    std::unique_ptr<UpsampleNearest2dFwd> UpsampleNearest2dFwd::create(Context &ctx,Shape const &ishape,Shape const &oshape,double s_h,double s_w,DataType dt)
    {
        std::unique_ptr<UpsampleNearest2dFwd> r(new UpsampleNearest2dImpl(ctx,ishape,oshape,s_h,s_w,dt));
        return r;
    }

} // core_ops
} // dlprim
