///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/core/common.hpp>
#include <dlprim/core/mypointwise.hpp>
#include <dlprim/gpu/program_cache.hpp>

#include <iostream>
#include <sstream>

namespace dlprim {
namespace core {

    template<int size>
    struct CLShape {
        cl_ulong s[size];
    };

    template<int size>
    void bind_cl_shape(cl::Kernel &k,int &p,Shape const &s)
    {
        CLShape<size> cl_s;
        for(int i=0;i<size;i++)
            cl_s.s[size-1-i]=s.size()-1-i>=0?s[s.size()-1-i]:1;
        k.setArg(p++,cl_s);
    }

     void bind_shape(cl::Kernel &k,int &p,int DIMS,Shape const &s)
    {
        switch(DIMS) {
        case 1: bind_cl_shape<1>(k,p,s); return;
        case 2: bind_cl_shape<2>(k,p,s); return;
        case 3: bind_cl_shape<3>(k,p,s); return;
        case 4: bind_cl_shape<4>(k,p,s); return;
        case 5: bind_cl_shape<5>(k,p,s); return;
        case 6: bind_cl_shape<6>(k,p,s); return;
        case 7: bind_cl_shape<7>(k,p,s); return;
        case 8: bind_cl_shape<8>(k,p,s); return;
        default:
            {
                std::ostringstream ss;
                ss << "Shape isn't valid " << s;
                throw ValidationError(ss.str());
            }
        }
    }

    void mypointwise_operation_broadcast(Tensor &x,Tensor &y,double w,std::string const &code,ExecutionContext const &e)
    {
    
        DLPRIM_CHECK(x.shape().total_size());
        DLPRIM_CHECK(y.shape().total_size());
        
        Context ctx(e);

        std::ostringstream params,load,save;
        load<<"dtype x0=x[x_offset+index];";
        params<<"__global dtype const *x"<< ", ulong x_offset ";
        std::string type = data_type_to_opencl_type(y.dtype());
        params<<",__global "<<type<< " *y"<< ", ulong y_offset ";

        params<<",const double w";
        load<<type<<" y0;";
        save<<"y[y_offset+index]=y0;";

        cl::Program const &prog = gpu::Cache::instance().get_program(ctx,  "mypointwise_broadcast",
                                                                           "dtype",data_type_to_opencl_type(x.dtype()),
                                                                           "#PARAMS",params.str(),
                                                                           "#LOAD",load.str(),
                                                                           "#CALC",code,
                                                                           "#SAVE",save.str()
                                                                           );
        cl::Kernel k(prog,"myexec");
        int p=0;
        x.set_arg(k,p);
        y.set_arg(k,p);
        k.setArg(p++,w);

        e.queue().enqueueNDRangeKernel(k,cl::NullRange,y.shape().total_size(),cl::NullRange,e.events(),e.event("mypointwise_exec_broadcast"));
    }
} // core
} // dlprim

