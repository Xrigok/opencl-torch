///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/bmm.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <iostream>

namespace dlprim {
namespace core {
    class BatchMatMulImpl : public BMMForward {
    public:
        virtual ~BatchMatMulImpl() {}

        BatchMatMulImpl(Context &ctx,Shape const &s1,Shape const &s2,DataType dtype)
        {
            dt_ = dtype;
            DLPRIM_CHECK(dtype == float_data);
            int ps=std::min(int(s1[1]),std::min(int(s2[1]),int(s2[2])));
            int ls;
            if(ps>=128){
                ls=64;
            }
            else if(ps>=64){
                ls=32;
            }
            else if(ps>=32){
                ls=16;
            }
            else if(ps>=16){
                ls=8;
            }
            else{
                ls=1;
            }
            local_size={ls,ls};

            int gs1=(s1[1]+ls-1)/ls*ls;
            int gs2=(s2[2]+ls-1)/ls*ls;
            global_size={gs1,gs2,int(s1[0])};

            cl::Program const &utils = gpu::Cache::instance().get_program(ctx,"bmm","P",ls);
            
            forward_ = cl::Kernel(utils,"bmm_forward");
        }

        virtual void enqueue(Tensor &in,Tensor &weight,Tensor &out,ExecutionContext const &e)
        {

            int p = 0;
            in.set_arg(forward_,p);
            weight.set_arg(forward_,p);
            out.set_arg(forward_,p);

            forward_.setArg(p++,int(in.shape()[1]));
            forward_.setArg(p++,int(weight.shape()[2]));
            forward_.setArg(p++,int(in.shape()[2]));

            auto e1=e.generate_series_context(0,2);
            auto e2=e.generate_series_context(1,2);
            e.queue().enqueueNDRangeKernel(forward_,cl::NullRange,cl::NDRange(global_size),cl::NDRange(local_size),e1.events(),e1.event("batch_MatMul"));

        }
    
    private:
        cl::NDRange global_size;
        cl::NDRange local_size;

        DataType dt_;
        cl::Kernel forward_;

        Tensor null_;
    };

    std::unique_ptr<BMMForward> BMMForward::create(Context &ctx,Shape const &s1,Shape const &s2,DataType dt)
    {
        std::unique_ptr<BMMForward> r(new BatchMatMulImpl(ctx,s1,s2,dt));
        return r;
    }

} // core_ops
} // dlprim
