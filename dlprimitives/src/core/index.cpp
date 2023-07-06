///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/index.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <iostream>

namespace dlprim {
namespace core {
    class IndexSelectImpl : public IndexSelectForward {
    public:
        virtual ~IndexSelectImpl() {}
        IndexSelectImpl(Context &ctx,long M2,long M3,long D1,
                        long D2,long OD1,long L,size_t out_size,DataType dtype){
            dt_ = dtype;

            local_size=256;
            global_size=(out_size+255)/256*256;

            cl::Program const &utils = gpu::Cache::instance().get_program(ctx,"index_select",
                                                                                "dtype", data_type_to_opencl_type(dtype),
                                                                                "M2",M2,
                                                                                "M3",M3,
                                                                                "D1",D1,
                                                                                "D2",D2,
                                                                                "OD1",OD1,
                                                                                "L",L,
                                                                                "XL",L*M2,
                                                                                "ls",256,
                                                                                "out_size",out_size);
            
            forward_ = cl::Kernel(utils,"index_select");

        }

        virtual void enqueue(Tensor &input,Tensor &XX,Tensor &DS,Tensor &out,ExecutionContext const &e)
        {

            int p = 0;

            input.set_arg(forward_,p);
            out.set_arg(forward_,p);
            DS.set_arg(forward_,p);
            XX.set_arg(forward_,p);


            auto e1=e.generate_series_context(0,2);
            auto e2=e.generate_series_context(1,2);
            e.queue().enqueueNDRangeKernel(forward_,cl::NullRange,global_size,local_size,e1.events(),e1.event("index_select"));

        }
    
    private:
        cl::NDRange global_size;
        cl::NDRange local_size;

        DataType dt_;
        cl::Kernel forward_;

        dlprim::Tensor XX,DS;

        Tensor null_;
    };
    std::unique_ptr<IndexSelectForward> IndexSelectForward::create(Context &ctx,long M2,long M3,long D1,
                                                            long D2,long OD1,long L,size_t out_size,DataType dt)
    {
        std::unique_ptr<IndexSelectForward> r(new IndexSelectImpl(ctx,M2,M3,D1,D2,OD1,L,out_size,dt));
        return r;
    }


    class IndexPutImpl : public IndexPutForward {
    public:
        virtual ~IndexPutImpl() {}
        IndexPutImpl(Context &ctx,long D1, long DA,long DC,long M2,long L,DataType dtype){
            dt_ = dtype;
            DLPRIM_CHECK(dtype == float_data);
            local_size={8,8,4};

            global_size={(DA+7)/8*8,(L+7)/8*8,(DC+3)/4*4};

            cl::Program const &utils = gpu::Cache::instance().get_program(ctx,"index_put",
                                                                                "D1",D1,
                                                                                "M2",M2,
                                                                                "DA",DA,
                                                                                "DB",L,
                                                                                "DC",DC,
                                                                                "L",L,
                                                                                "ls2",8,
                                                                                "ls3",4,
                                                                                "XL",L*M2,
                                                                                "ls",256);
            forward_ = cl::Kernel(utils,"index_put");

        }

        virtual void enqueue(Tensor &input,Tensor &XX,Tensor &DS,Tensor &value,ExecutionContext const &e)
        {

            int p = 0;

            input.set_arg(forward_,p);
            value.set_arg(forward_,p);
            DS.set_arg(forward_,p);
            XX.set_arg(forward_,p);


            auto e1=e.generate_series_context(0,2);
            auto e2=e.generate_series_context(1,2);
            e.queue().enqueueNDRangeKernel(forward_,cl::NullRange,global_size,local_size,e1.events(),e1.event("index_select"));

        }
    
    private:
        cl::NDRange global_size;
        cl::NDRange local_size;

        DataType dt_;
        cl::Kernel forward_;

        dlprim::Tensor XX,DS;

        Tensor null_;
    };
    std::unique_ptr<IndexPutForward> IndexPutForward::create(Context &ctx,long D1,long DA,long DC,long M2,long L,DataType dt)
    {
        std::unique_ptr<IndexPutForward> r(new IndexPutImpl(ctx,D1,DA,DC,M2,L,dt));
        return r;
    }

} // core_ops
} // dlprim
