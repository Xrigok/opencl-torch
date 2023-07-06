///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/sort.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <iostream>

namespace dlprim {
namespace core {
    class SortImpl : public SortForward {
    public:
        virtual ~SortImpl() {}

        SortImpl(Context &ctx,size_t M1,size_t M2,size_t M3,DataType dtype,bool stable,bool descending)
        {
            dt_ = dtype;
            int ls;
            if(M2>=128){
                ls=16;
            }
            else if(M2>=64){
                ls=8;
            }
            else if(M2>=32){
                ls=4;
            }
            else if(M2>=16){
                ls=2;
            }
            else{
                ls=1;
            }
            local_size={1,ls,1};

            global_size={M1,ls,M3};
            
            std::string code;
            if(stable){
                code="compareStable(a[left1],idxs[left1],a[right1],idxs[right1])";
            }
            else{
                code="compare(a[left1],a[right1])";
            }
            cl::Program const &utils = gpu::Cache::instance().get_program(ctx,"argsort","LS",ls,
                                                                                    "M2",M2,
                                                                                    "M3",M3,
                                                                                    "descending",descending,
                                                                                    "#CODE",code);
            
            forward_ = cl::Kernel(utils,"quick_sort");
        }

        virtual void enqueue(Tensor &x,Tensor &y,Tensor &indx,ExecutionContext const &e)
        {

            int p = 0;
            x.set_arg(forward_,p);
            y.set_arg(forward_,p);
            indx.set_arg(forward_,p);

            auto e1=e.generate_series_context(0,2);
            auto e2=e.generate_series_context(1,2);
            e.queue().enqueueNDRangeKernel(forward_,cl::NullRange,cl::NDRange(global_size),cl::NDRange(local_size),e1.events(),e1.event("argsort"));

        }
    
    private:
        cl::NDRange global_size;
        cl::NDRange local_size;

        DataType dt_;
        cl::Kernel forward_;

        Tensor null_;
    };

    std::unique_ptr<SortForward> SortForward::create(Context &ctx,size_t M1,size_t M2,size_t M3,DataType dt,bool stable,bool descending)
    {
        std::unique_ptr<SortForward> r(new SortImpl(ctx,M1,M2,M3,dt,stable,descending));
        return r;
    }

} // core_ops
} // dlprim
