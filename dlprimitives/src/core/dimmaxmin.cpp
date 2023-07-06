///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/dim_maxmin.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <iostream>

namespace dlprim {
namespace core {
    class DimMaxMinImpl : public DMMForward {
    public:
        virtual ~DimMaxMinImpl() {}

        DimMaxMinImpl(Context &ctx,size_t M1,size_t M2,size_t M3,DataType dtype,std::string method="max")
        {
            dt_ = dtype;
            DLPRIM_CHECK(dtype == float_data);
            int ls;
            if(M2>=128){
                ls=32;
            }
            else if(M2>=64){
                ls=16;
            }
            else if(M2>=32){
                ls=8;
            }
            else if(M2>=16){
                ls=4;
            }
            else if(M2>=8){
                ls=2;
            }
            else{
                ls=1;
            }
            local_size={1,ls,1};

            global_size={M1,ls,M3};
            std::string code1,code2;
            if(method=="max"){
                code1="cur_v>v[ly]";
                code2="v[ly]<v[ly+mid]";
            }
            else{
                code1="cur_v<v[ly]";
                code2="v[ly]>v[ly+mid]";
            }
            
            cl::Program const &utils = gpu::Cache::instance().get_program(ctx,"dim_maxmin","P",ls,
                                                                                    "L1",M2,
                                                                                    "L2",M3,
                                                                                    "#CALC1",code1,
                                                                                    "#CALC2",code2);
            
            forward_ = cl::Kernel(utils,"dim_max_min");
        }

        virtual void enqueue(Tensor &x,Tensor &y,Tensor &indx,ExecutionContext const &e)
        {

            int p = 0;
            x.set_arg(forward_,p);
            y.set_arg(forward_,p);
            indx.set_arg(forward_,p);

            auto e1=e.generate_series_context(0,2);
            auto e2=e.generate_series_context(1,2);
            e.queue().enqueueNDRangeKernel(forward_,cl::NullRange,cl::NDRange(global_size),cl::NDRange(local_size),e1.events(),e1.event("dim_max_min"));

        }
    
    private:
        cl::NDRange global_size;
        cl::NDRange local_size;

        DataType dt_;
        cl::Kernel forward_;

        Tensor null_;
    };

    std::unique_ptr<DMMForward> DMMForward::create(Context &ctx,size_t M1,size_t M2,size_t M3,DataType dt,std::string method)
    {
        std::unique_ptr<DMMForward> r(new DimMaxMinImpl(ctx,M1,M2,M3,dt,method));
        return r;
    }

} // core_ops
} // dlprim
