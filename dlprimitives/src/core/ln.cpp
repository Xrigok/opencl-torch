///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/ln.hpp>
#include <dlprim/gpu/program_cache.hpp>

namespace dlprim {
namespace core {
    class LayerNormImpl : public LayerNormFwdBwd {
    public:
        virtual ~LayerNormImpl() {}

        LayerNormImpl(Context &ctx,Shape const &s,long const ns,DataType dtype)
        {
            dt_ = dtype;
            DLPRIM_CHECK(dtype == float_data);
            if(ns>=512){
                local_size=256;
            }
            else if(ns>=256){
                local_size=128;
            }
            else if(ns>=128){
                local_size=64;
            }
            else if(ns>=64){
                local_size=32;
            }
            else if(ns>=32){
                local_size=16;
            }
            else if(ns>=16){
                local_size=8;
            }
            else{
                local_size=1;
            }
            
            int size=s.total_size();
            group_size=size/ns*local_size;

            cl::Program const &utils = gpu::Cache::instance().get_program(ctx,"ln_utils",
                                                                    "input_size",size,
                                                                    "norm_size",ns,
                                                                    "local_size",local_size);
            
            forward_direct = cl::Kernel(utils,"forward_direct");
            forward_affine = cl::Kernel(utils,"forward_affine");
        }

        virtual void enqueue_forward_direct(Tensor &x,Tensor &y,float eps,
                                            ExecutionContext const &e)
        {

            int p = 0;
            x.set_arg(forward_direct,p);
            y.set_arg(forward_direct,p);

            auto e1=e.generate_series_context(0,2);
            auto e2=e.generate_series_context(1,2);
            e.queue().enqueueNDRangeKernel(forward_direct,cl::NullRange,cl::NDRange(group_size),cl::NDRange(local_size),e1.events(),e1.event("not_gamma_beta_to_ln"));

        }

        virtual void enqueue_forward_affine(Tensor &x,Tensor &y,
                                            Tensor &gamma,Tensor &beta,
                                            float eps,
                                            ExecutionContext const &e)
        {
            int p = 0;
            x.set_arg(forward_affine,p);
            y.set_arg(forward_affine,p);
            gamma.set_arg(forward_affine,p);
            beta.set_arg(forward_affine,p);

            auto e1=e.generate_series_context(0,2);
            auto e2=e.generate_series_context(1,2);
            e.queue().enqueueNDRangeKernel(forward_affine,cl::NullRange,cl::NDRange(group_size),cl::NDRange(local_size),e1.events(),e1.event("gamma_beta_to_ln"));

        }

    
    private:
        int group_size;
        int local_size;

        DataType dt_;
        cl::Kernel forward_direct;
        cl::Kernel forward_affine;

        Tensor null_;
    };

    std::unique_ptr<LayerNormFwdBwd> LayerNormFwdBwd::create(Context &ctx,Shape const &s,long const &ns,DataType dt)//ns:norm_size
    {
        std::unique_ptr<LayerNormFwdBwd> r(new LayerNormImpl(ctx,s,ns,dt));
        return r;
    }

} // core_ops
} // dlprim
