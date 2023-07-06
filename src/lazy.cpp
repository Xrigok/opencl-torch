#include "CLTensor.h"
#include "utils.h"

#include <dlprim/core/util.hpp>
#include <dlprim/core/bmm.hpp>
#include <dlprim/core/loss.hpp>

#include <dlprim/random.hpp>
#include <iostream>
namespace ptdlprim {

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;

    Tensor & bmm_out(Tensor const& self, Tensor const& weight,Tensor &out)
    {
        GUARD;

        Tensor self_c = self.contiguous();
        dlprim::Tensor X1 = todp(self_c);

        Tensor weight_c = weight.contiguous();
        dlprim::Tensor X2 = todp(weight_c);

        dlprim::Tensor result = todp(out);

        dlprim::ExecutionContext q=getExecutionContext(self);
        dlprim::Context ctx(q);

        auto x1shape=X1.shape();
        auto x2shape=X2.shape();
        
        TORCH_CHECK(x1shape[0] == x2shape[0],"the dim 0 of x1 diffierence with x2, please check");
        TORCH_CHECK(x1shape[2] == x2shape[1],"the dim 1/2 of x1 diffierence with dim 2/1 of x2, please check");

        auto bmm = dlprim::core::BMMForward::create(ctx,x1shape,x2shape,X1.dtype());
        bmm->enqueue(X1,X2,result,q);
        
        sync_if_needed(self.device());
        return out;
    }

    Tensor & _softmax(Tensor const& self,int64_t dim,bool half_to_float,Tensor &out){
        GUARD;
        Tensor self_c = self.contiguous();
        dlprim::Tensor x=todp(self_c);
        TORCH_CHECK(dim==-1||dim==x.shape().size()-1,"Only case lastdim is supported currently");

        auto xshape=x.shape();
        int ns=1;
        for(int i=0;i<xshape.size()-1;++i){
            ns*=xshape[i];
        }

        x.reshape(dlprim::Shape(ns,xshape[xshape.size()-1]));


        
        dlprim::Tensor y=todp(out);
        y.reshape(dlprim::Shape(ns,xshape[xshape.size()-1]));
        dlprim::core::softmax_forward(x,y,false,getExecutionContext(self));
        sync_if_needed(self.device());

        return out;
    }

} // namespace
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::bmm.out",&ptdlprim::bmm_out);
    m.impl("aten::_softmax.out",&ptdlprim::_softmax);
    
    
}

