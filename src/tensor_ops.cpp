#include "CLTensor.h"
#include "utils.h"
#include <ATen/native/CPUFallback.h>

#include <dlprim/core/util.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/core/index.hpp>
#include <dlprim/core/dim_maxmin.hpp>
#include <dlprim/core/sort.hpp>
#include <dlprim/core/flip.hpp>

#include <iostream>
namespace ptdlprim {

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;


    using torch::Tensor;

    torch::Tensor allocate_empty(torch::IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> /*layout*/, c10::optional<Device> device, c10::optional<bool> /*pin_memory*/, c10::optional<MemoryFormat> /*memory_format*/)
    {
        GUARD;
        c10::ScalarType st = dtype ? *dtype : c10::kFloat; 
        c10::Device dev = device ? *device : Device(OpenCLDeviceType,0);
        return ptdlprim::new_ocl_tensor(size,dev,st);
    }

    /// "aten::empty_strided"
    Tensor empty_strided(torch::IntArrayRef size, torch::IntArrayRef /*stride*/, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) 
    {
        GUARD;
        return allocate_empty(size,dtype,layout,device,pin_memory,c10::nullopt);
    }

    torch::Tensor _reshape_alias(const Tensor & self, c10::IntArrayRef size, c10::IntArrayRef stride)
    {
        GUARD;
        torch::Tensor data = at::alias(self);
        data.getIntrusivePtr()->set_sizes_and_strides(size,stride);
        return data;
    }

    Tensor view(const Tensor & self, IntArrayRef size)
    {
        GUARD;
        torch::Tensor data=at::alias(self);
        TORCH_CHECK(data.is_contiguous(),"View imlemented on contiguous array");
        std::vector<int64_t> v(size.begin(),size.end());
        int64_t total=1,index=-1;
        for(unsigned i=0;i<v.size();i++) {
            if(v[i] == -1) {
                TORCH_CHECK(index==-1,"Must be unique -1");
                index=i;
            }
            else {
                total *= v[i];
            }
        }
        if(index != -1) {
            TORCH_CHECK(self.numel() % total == 0);
            v[index] = self.numel() / total;
        }
        else {
            TORCH_CHECK(total == self.numel());
        }
        c10::IntArrayRef new_size(v.data(),v.size());
        data.getIntrusivePtr()->set_sizes_contiguous(new_size);
        return data;
    }

    static Tensor make_contiguous_as_target_type(Tensor const &self,Tensor const &dst)
    {
        GUARD;
        Tensor c_src = self;
        if(self.dtype() != dst.dtype() || !self.is_contiguous()) {
            TensorOptions options = TensorOptions().dtype(dst.dtype()).memory_format(MemoryFormat::Contiguous);
            Tensor temp = at::empty_like(c_src,options);
            temp.copy_(c_src);
            c_src = temp;
        }
        return c_src;
    }

    Tensor _copy_from(const Tensor & self, const Tensor & dst, bool /*non_blocking*/)
    {
        GUARD;
        if(self.numel() == 0){
            return self;
        }
        if(dst.device().type() == c10::DeviceType::CPU && self.device().type() == OpenCLDeviceType) {
            Tensor c_src = make_contiguous_as_target_type(self,dst);

            dlprim::Tensor t(todp(c_src));
            auto ec = getExecutionContext(self);
            if(dst.is_contiguous()) {
                void *ptr = dst.data_ptr();
                t.to_host(ec,ptr);
            }
            else {
                TensorOptions options = TensorOptions().memory_format(MemoryFormat::Contiguous);
                Tensor dst_c = at::empty_like(dst,options);
                void *ptr = dst_c.data_ptr();
                t.to_host(ec,ptr);
                dst.copy_(dst_c);
            }
        }
        else if(self.device().type() == c10::DeviceType::CPU && dst.device().type() == OpenCLDeviceType) {
            Tensor c_src = make_contiguous_as_target_type(self,dst);
            auto ec = getExecutionContext(dst);
            if(dst.is_contiguous()) {
                dlprim::Tensor t(todp(dst));
                t.to_device(ec,c_src.data_ptr());
            }
            else {
                TensorOptions options = TensorOptions().memory_format(MemoryFormat::Contiguous);
                Tensor temp = at::empty_like(dst,options);
                dlprim::Tensor t(todp(temp));
                t.to_device(ec,c_src.data_ptr());
                dst.copy_(temp);
            }
        }
        else if(self.device().type() == OpenCLDeviceType && dst.device().type() == OpenCLDeviceType) {
            if(self.is_contiguous() && dst.is_contiguous()) {
                dlprim::core::pointwise_operation_broadcast({todp(self)},{todp(dst)},{},"y0=x0;",getExecutionContext(self.device()));
            }
            else {
                auto src_sizes  = self.sizes();
                auto src_stride = self.strides();
                auto src_offset = self.storage_offset();
                auto tgt_sizes  = dst.sizes();
                auto tgt_stride = dst.strides();
                auto tgt_offset = dst.storage_offset();
                TORCH_CHECK(src_sizes == tgt_sizes);
                dlprim::Shape shape=dlprim::Shape::from_range(src_sizes.begin(),src_sizes.end());
                dlprim::Shape src_std=dlprim::Shape::from_range(src_stride.begin(),src_stride.end());
                dlprim::Shape tgt_std=dlprim::Shape::from_range(tgt_stride.begin(),tgt_stride.end());
                dlprim::core::copy_strided(shape,buffer_from_tensor(self),src_offset,src_std,
                                                 buffer_from_tensor(dst), tgt_offset,tgt_std,
                                                 todp(self.dtype()),
                                                 todp(dst.dtype()),
                                                 getExecutionContext(self.device()));
            }
            sync_if_needed(self.device());
        }
        else {
            throw std::runtime_error("OpenCL supports copy to CPU backend only");
        }
        return self;
    }

    Tensor &fill_(Tensor &self, const c10::Scalar &value)
    {
        GUARD;
        if(self.numel() == 0){
            return self;
        }
        dlprim::Tensor t(todp(self));
        auto q = getExecutionContext(self);
        dlprim::core::fill_tensor(t,value.to<double>(),q);
        sync_if_needed(self.device());
        return self;
    }
    
    Tensor &zero_(Tensor &self)
    {
        GUARD;
        dlprim::Tensor t(todp(self));
        dlprim::core::fill_tensor(t,0.0,getExecutionContext(self));
        return self;
    }

    Tensor zeros(IntArrayRef size,c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory)
    {
        GUARD;
        return allocate_empty(size,dtype,layout,device,pin_memory,c10::nullopt);
    }

    Tensor as_strided(const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset)
    {
        GUARD;
        Tensor result = at::alias(self);
        result.getIntrusivePtr()->set_sizes_and_strides(size,stride);
        if(storage_offset)
            result.getIntrusivePtr()->set_storage_offset(*storage_offset);
        return result;

    }

    // {"schema": "aten::_local_scalar_dense(Tensor self) -> Scalar", "dispatch": "True", "default": "False"}
    Scalar _local_scalar_dense(const Tensor & self)
    {
        GUARD;
        TORCH_CHECK(self.numel()==1);
        dlprim::Tensor x=todp(self);
        union {
            float f;
            double d;
            int8_t i8;
            uint8_t u8;
            int16_t i16;
            uint16_t u16;
            int32_t i32;
            uint32_t u32;
            int64_t i64;
            uint64_t u64;
            char data[16];
        } data;
        x.to_host(getExecutionContext(self),data.data);
        switch(x.dtype()) {
        case dlprim::float_data:    return data.f;
        case dlprim::double_data:   return data.d;
        case dlprim::int8_data:     return data.i8;
        case dlprim::uint8_data:    return data.u8;
        case dlprim::int16_data:    return data.i16;
        case dlprim::uint16_data:   return data.u16;
        case dlprim::int32_data:    return (int64_t)data.i32;
        case dlprim::uint32_data:   return (int64_t)data.u32;
        case dlprim::int64_data:    return (int64_t)data.i64;
        case dlprim::uint64_data:   return (int64_t)data.u64;
        default:
            TORCH_CHECK(!"Not implemented dtype","Not implemented data type");
        }
    }

    template<typename E,typename M>
    size_t select_impl_by(E *p,M *m,size_t n)
    {
        size_t N = 0;
        for(size_t i=0;i<n;i++) {
            if(m[i]) {
                p[N] = p[i];
                N++;
            }
        }
        return N;
    }

    template<typename T>
    size_t select_impl(T *mask,dlprim::Tensor &/*m*/,dlprim::Tensor &v)
    {
        void *p=v.host_data();
        switch(dlprim::size_of_data_type(v.dtype())) {
        case 1: return select_impl_by(static_cast<int8_t  *>(p),mask,v.shape().total_size());
        case 2: return select_impl_by(static_cast<int16_t *>(p),mask,v.shape().total_size());
        case 4: return select_impl_by(static_cast<int32_t *>(p),mask,v.shape().total_size());
        case 8: return select_impl_by(static_cast<int64_t *>(p),mask,v.shape().total_size());
        default:
            TORCH_CHECK(!"Invalid sizeof");
            return 0;
        }
    }
    
    // {"schema": "aten::masked_select(Tensor self, Tensor mask) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor masked_select(const Tensor & self, const Tensor & mask)
    {
        GUARD;
        if(mask.numel() == 0){
            return self;
        }
        Tensor self_c = self.contiguous();
        Tensor mask_c = mask.contiguous();
        dlprim::Tensor x = todp(self_c);
        dlprim::Tensor m = todp(mask_c);
        TORCH_CHECK(x.shape() == m.shape(),"Broadasting is not implemented in masked_select yet");
        auto ec = getExecutionContext(self);
        x.to_host(ec);
        m.to_host(ec);
        size_t N = 0;
        switch(m.dtype()) {
        case dlprim::float_data:
            N = select_impl(m.data<float>(),m,x);
            break;
        case dlprim::double_data:
            N = select_impl(m.data<double>(),m,x);
            break;
        case dlprim::int8_data:
            N = select_impl(m.data<int8_t>(),m,x);
            break;
        case dlprim::uint8_data:
            N = select_impl(m.data<uint8_t>(),m,x);
            break;
        case dlprim::int16_data:
            N = select_impl(m.data<int16_t>(),m,x);
            break;
        case dlprim::uint16_data:
            N = select_impl(m.data<uint16_t>(),m,x);
            break;
        case dlprim::int32_data:
            N = select_impl(m.data<int32_t>(),m,x);
            break;
        case dlprim::uint32_data:
            N = select_impl(m.data<uint32_t>(),m,x);
            break;
        case dlprim::int64_data:
            N = select_impl(m.data<int64_t>(),m,x);
            break;
        case dlprim::uint64_data:
            N = select_impl(m.data<uint64_t>(),m,x);
            break;
        default:
            TORCH_CHECK(!"Not implemented dtype","Not implemented");
        }
        Tensor res=new_tensor_as(dlprim::Shape(N),self);
        if(N > 0) {
            dlprim::Tensor y=todp(res);
            y.to_device(getExecutionContext(self),x.host_data());
        }
        sync_if_needed(self.device());
        return res;
    }

    void fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack)
    {
      TORCH_WARN("The operator '", op.schema().operator_name(), "' is not currently ",
                 "supported on the ocl backend. Please open an issue at for requesting support "
                 "https://github.com/artyom-beilis/pytorch_dlprim/issues");
      native::cpu_fallback(op, stack);
    }

    Tensor arange(c10::Scalar const& end,c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory){
        GUARD;

        int e=end.to<double>();
        c10::ScalarType st = dtype ? *dtype : c10::kLong; 
        c10::Device dev = device ? *device : Device(OpenCLDeviceType,0);

        Tensor out=new_ocl_tensor(e,dev,st);
        dlprim::Tensor Y = todp(out);
        auto q = getExecutionContext(out);
        dlprim::core::pointwise_operation({},{Y},{},"y0 =index;",q);
        sync_if_needed(out.device());
        return out;
    }

    void decompose_indices(c10::List<c10::optional<at::Tensor> > const& indices,std::vector<long> &XX,std::vector<long> &DS,
                           dlprim::Shape const &xs,long &M2, long &M3, long &DA,long &D1,long &D2,long &OD1, long &L,
                           dlprim::ExecutionContext q){
        long M1=0;
        M2=0;
        M3=0;
        D1=xs.total_size();
        D2=D1;
        OD1=1;
        L=0;
        DA=1;

        for(int i=0;i<indices.size();++i){
            auto indx = *indices.get(i);
            if(indx.dtype()!=c10::kBool&&indx.dtype()!=c10::kLong){
                M1+=1;
                D1/=xs[i];
                D2=D1;
                DA*=xs[i];
            }
            else if(indx.dtype()==c10::kLong){
                M2+=1;
                D2/=xs[i];
                
                Tensor tmp_indx_c=indx.contiguous();
                dlprim::Tensor tmp_indx=todp(tmp_indx_c);
                dlprim::Shape tmp_indx_shape=tmp_indx.shape();
                L=tmp_indx_shape.total_size();

                tmp_indx.to_host(q);
                void *p = tmp_indx.host_data();
                auto v = static_cast<int64_t *>(p);
                for(long i=0;i<L;++i){
                    XX.emplace_back(*(v++));
                }
                
                DS.emplace_back(xs[i]);
            }
            else if(indx.dtype()==c10::kBool){
                
                Tensor out = nonzero(indx);
                Tensor out_T=out.transpose(0,1);
                Tensor out_c=out_T.contiguous();

                dlprim::Tensor tmp=todp(out_c);
                L=tmp.shape()[1];
                for(int j=0;j<tmp.shape()[0];++j){
                    D2/=xs[i+j];
                    DS.emplace_back(xs[i+j]);
                    M2+=1;
                }

                tmp.to_host(q);
                void *p = tmp.host_data();
                auto v = static_cast<int64_t *>(p);

                for(int i=0;i<tmp.shape().total_size();++i){
                    XX.emplace_back(*(v++));
                }
            }
        }
        M3=xs.size()-M1-M2;

        OD1=L*D2;
    }

    template<typename T>
    void index_Tensor_out_impl(T *A, T *B, size_t D1, size_t D2, size_t OD1, size_t DA, size_t L, size_t M2, size_t M3, std::vector<long> &XX, std::vector<long> &DS){
        
        for(size_t da = 0; da < DA; ++da){
            for(size_t l = 0; l < L; ++l){
                size_t offset = da * D1;
                size_t last_l = D2;
                for(int m2 = M2 - 1; m2 >= 0; --m2){
                    size_t p=XX[m2*L+l];
                    offset += p*last_l;
                    last_l*=DS[m2];
                }
                for(size_t j = 0; j <D2; ++j){
                    B[da*OD1 + l*D2 + j] = A[offset +j];
                }
            }
        }

    }

    Tensor & index_Tensor_out(Tensor const& self,c10::List<c10::optional<at::Tensor> > const& indices,Tensor &out){
        auto start = std::chrono::high_resolution_clock::now();
        GUARD;
        if(out.numel() == 0){
            return out;
        }
        Tensor self_c=self.contiguous();
        dlprim::Tensor x=todp(self_c);
        dlprim::Tensor y=todp(out);
        dlprim::Shape x_shape=x.shape();
        dlprim::ExecutionContext q=getExecutionContext(self);
        dlprim::Context ctx(q);

        size_t out_size=y.shape().total_size();
        long M1,M2,M3,D1,D2,OD1,L;
        long DA;
        std::vector<long>XX;
        std::vector<long>DS;

        decompose_indices(indices,XX,DS,x_shape,M2,M3,DA,D1,D2,OD1,L,q);

        x.to_host(q);
        void *p1 = x.host_data();
        y.to_host(q);
        void *p2 = y.host_data();
        
        switch(x.dtype()) {
            case dlprim::float_data:
                index_Tensor_out_impl(static_cast<float *>(p1),static_cast<float *>(p2),D1,D2,OD1,DA,L,M2,M3,XX,DS);
                break;
            case dlprim::double_data:

                index_Tensor_out_impl(static_cast<double *>(p1),static_cast<double *>(p2),D1,D2,OD1,DA,L,M2,M3,XX,DS);
                break;
            case dlprim::int8_data:
                index_Tensor_out_impl(static_cast<int8_t *>(p1),static_cast<int8_t *>(p2),D1,D2,OD1,DA,L,M2,M3,XX,DS);
                break;
             case dlprim::uint8_data:
                index_Tensor_out_impl(static_cast<uint8_t *>(p1),static_cast<uint8_t *>(p2),D1,D2,OD1,DA,L,M2,M3,XX,DS);
                break;
            case dlprim::int16_data:
                index_Tensor_out_impl(static_cast<int16_t *>(p1),static_cast<int16_t *>(p2),D1,D2,OD1,DA,L,M2,M3,XX,DS);
                break;
            case dlprim::int32_data:
                index_Tensor_out_impl(static_cast<int32_t *>(p1),static_cast<int32_t *>(p2),D1,D2,OD1,DA,L,M2,M3,XX,DS);
                break;
            case dlprim::int64_data:
                index_Tensor_out_impl(static_cast<int64_t *>(p1),static_cast<int64_t *>(p2),D1,D2,OD1,DA,L,M2,M3,XX,DS);
                break;
            default:
                TORCH_CHECK(!"Not implemented dtype","Not implemented");
        }
        y.to_device(q);
        /*
        dlprim::Tensor new_XX(ctx,dlprim::Shape(XX.size()),dlprim::int64_data);
        new_XX.to_device(q,XX.data());
        dlprim::Tensor new_DS(ctx,dlprim::Shape(DS.size()),dlprim::int64_data);
        new_DS.to_device(q,DS.data());
        
        auto index_select = dlprim::core::IndexSelectForward::create(
                        ctx,M2,M3,D1,D2,OD1,L,out_size,
                        x.dtype()
                    );
        index_select->enqueue(x,new_XX,new_DS,y,q); 
        */
        sync_if_needed(self.device());
        return out;
    }

    Tensor & index_put_impl(Tensor& self,c10::List<c10::optional<at::Tensor> > const& indices,Tensor const& values,bool accumulate, bool unsafe){
        GUARD;
        TORCH_CHECK(accumulate==false,"Can have affince or nounsopport accumulate")
        TORCH_CHECK(unsafe==false,"Can have affince or nounsopport unsafe")
        if(self.numel() == 0){
            return self;
        }
        Tensor self_c = self.contiguous();
        Tensor values_c = values.contiguous();

        dlprim::Tensor X = todp(self_c);
        dlprim::Tensor V = todp(values_c);
        dlprim::Shape x_shape=X.shape();
        dlprim::ExecutionContext q=getExecutionContext(self);
        dlprim::Context ctx(q);
        

        long DA,DC,M2,L;
        long M1,M3,D1,OD1;
        std::vector<long>XX;
        std::vector<long>DS;

        decompose_indices(indices,XX,DS,x_shape,M2,M3,DA,D1,DC,OD1,L,q);
        

        dlprim::Tensor new_XX(ctx,dlprim::Shape(XX.size()),dlprim::int64_data);
        new_XX.to_device(q,XX.data());
        dlprim::Tensor new_DS(ctx,dlprim::Shape(DS.size()),dlprim::int64_data);
        new_DS.to_device(q,DS.data());
       
        auto index_put = dlprim::core::IndexPutForward::create(
                        ctx,D1,DA,DC,M2,L,X.dtype()
                    );
        index_put->enqueue(X,new_XX,new_DS,V,q); 

        sync_if_needed(self.device());
        if(!self.is_contiguous()){
            self.copy_(self_c);
        }
        return self;
    }

    template<typename T>
    void nonzero_imp(T *p,size_t N,int &m,dlprim::Shape X_shape,std::vector<long>& indexs){
        for(long i=0;i<N;++i){
            if(*(p++)){
                long M1 = 1, M2=1;
                std::vector<long> tmp(m);
                for(long j=m-1;j>=0;--j){
                    M1*=X_shape[j];
                    tmp[j]=(i%M1)/M2;
                    M2*=X_shape[j];
                }
                indexs.insert(indexs.end(),tmp.begin(),tmp.end());
            }
        }
    }
    Tensor nonzero(Tensor const& self){
        GUARD;
        if(self.numel() == 0){
            Tensor out;
            return out;
        }
        Tensor self_c = self.contiguous();
        dlprim::Tensor X = todp(self_c);
        dlprim::Shape X_shape = X.shape();
        dlprim::ExecutionContext q=getExecutionContext(self);

        int m=X_shape.size();
        self_c=self_c.reshape(X_shape.total_size());
        std::vector<long> nonzero_element;

        X.to_host(q);
        void *p=X.host_data();
        switch(X.dtype()) {
            case dlprim::float_data:
                nonzero_imp(static_cast<float *>(p),X_shape.total_size(),m,X_shape,nonzero_element);
                break;
            case dlprim::double_data:
                nonzero_imp(static_cast<double *>(p),X_shape.total_size(),m,X_shape,nonzero_element);
                break;
            case dlprim::int8_data:
                nonzero_imp(static_cast<int8_t *>(p),X_shape.total_size(),m,X_shape,nonzero_element);
                break;
             case dlprim::uint8_data:
                nonzero_imp(static_cast<uint8_t *>(p),X_shape.total_size(),m,X_shape,nonzero_element);
                break;
            case dlprim::int16_data:
                nonzero_imp(static_cast<int16_t *>(p),X_shape.total_size(),m,X_shape,nonzero_element);
                break;
            case dlprim::int32_data:
                nonzero_imp(static_cast<int32_t *>(p),X_shape.total_size(),m,X_shape,nonzero_element);
                break;
            case dlprim::int64_data:
                nonzero_imp(static_cast<int64_t *>(p),X_shape.total_size(),m,X_shape,nonzero_element);
                break;
            default:
                TORCH_CHECK(!"Not implemented dtype","Not implemented");
        }
        
        dlprim::Shape s({nonzero_element.size()/m, m});
        int64_t shape[s.size()];
        for(int i=0;i<s.size();i++)
            shape[i]=s[i];
        torch::Tensor result = new_ocl_tensor(c10::IntArrayRef(shape,s.size()),
                                              self.device(),c10::kLong);

        if(nonzero_element.size()>0){
            dlprim::Tensor res = todp(result);
            res.to_device(q,nonzero_element.data());
        }
        sync_if_needed(self.device());
        return result;
    }

    ::std::tuple<Tensor&,Tensor&> max_dim_max(Tensor const& self,long dim, bool keepdim,Tensor & max,Tensor & max_values){
        GUARD;
        Tensor self_c = self.contiguous();
        dlprim::Tensor X = todp(self_c);

        Tensor max_c = max.contiguous();
        dlprim::Tensor Y = todp(max_c);

        Tensor max_values_c = max_values.contiguous();
        dlprim::Tensor Z = todp(max_values_c);

        dlprim::ExecutionContext q=getExecutionContext(self);
        dlprim::Context ctx(q);

        size_t M1 = 1, M2 = 1, M3 = 1;
        for(long i=0;i<X.shape().size();++i){
            if(i<dim){
                M1*=X.shape()[i];
            }
            else if(i>dim){
                M3*=X.shape()[i];
            }
            else{
                M2=X.shape()[i];
            }
        }

        std::string menthod="max";
        auto dim_max = dlprim::core::DMMForward::create(ctx,M1,M2,M3,X.dtype(),menthod);
        dim_max->enqueue(X,Y,Z,q);
        sync_if_needed(self.device());
        return std::tuple<Tensor&,Tensor&>(max_values,max);
    }

    ::std::tuple<at::Tensor&, at::Tensor&> sort_values_stable(Tensor const& self,c10::optional<bool> stable,long dim,bool descending,Tensor &values,Tensor &indices){
        GUARD;
        if(self.numel() == 0){
            return std::tuple<Tensor&,Tensor&>(values,indices);
        }
        Tensor self_c = self.contiguous();
        dlprim::Tensor X = todp(self_c);

        Tensor values_c = values.contiguous();
        dlprim::Tensor Y = todp(values);

        Tensor indices_c = indices.contiguous();
        dlprim::Tensor Z = todp(indices_c);
        
        dlprim::ExecutionContext q=getExecutionContext(self);
        dlprim::Context ctx(q);

        if(dim<0){
            dim+=X.shape().size();
        }
        size_t M1 = 1, M2 = 1, M3 = 1;
        for(long i=0;i<X.shape().size();++i){
            if(i<dim){
                M1*=X.shape()[i];
            }
            else if(i>dim){
                M3*=X.shape()[i];
            }
            else{
                M2=X.shape()[i];
            }
        }
        auto dim_max = dlprim::core::SortForward::create(ctx,M1,M2,M3,X.dtype(),*stable,descending);
        dim_max->enqueue(X,Y,Z,q);

        sync_if_needed(self.device());
        return std::tuple<Tensor&,Tensor&>(values_c,indices_c);
    }

    template<typename T>
    void unique2impl(T *p,T *origin, int N,c10::Device dev,c10::ScalarType type,bool return_inverse, bool return_counts,
                Tensor &out0,Tensor &out1, Tensor &out2,dlprim::ExecutionContext q){
        int len = 1;
        T* tmp = new T[N];
        for(int i=0;i<N;++i){
            tmp[i]=p[i];
        }
        if(N>1){
            auto res = tmp;
            auto first = tmp;
            auto last = tmp + N;
            while (++first != last){
                if (*res != *first){
                    *(++res)=*first;
                    len++;
                }
            }
        }
        out0=new_ocl_tensor(c10::IntArrayRef(len),dev,type);
        dlprim::Tensor Y0 = todp(out0);
        std::vector<T> out;
        for(int i=0;i<len;++i){
            out.emplace_back(tmp[i]);
        }
        Y0.to_device(q,out.data());

        if(return_inverse){
            out1=new_ocl_tensor(c10::IntArrayRef(N),dev,c10::kLong);
            dlprim::Tensor Y1 = todp(out1);
            std::unordered_map<T,long> index;
            std::vector<long> inverse;
            for(long i=0;i<len;++i){
                index[tmp[i]]=i;
            }
            for(int i=0;i<N;++i){
                inverse.emplace_back(index[origin[i]]);
            }
            Y1.to_device(q,inverse.data());
        }
        if(return_counts){
            out2=new_ocl_tensor(c10::IntArrayRef(len),dev,c10::kLong);
            dlprim::Tensor Y2 = todp(out2);
            std::vector<long> count;
            auto left = p;
            auto right = p;
            auto last = p + N;
            long cnt=1;
            while(++right!=last){
                if(*right!=*left){
                    count.emplace_back(cnt);
                    cnt=1;
                    left=right;
                }
                ++cnt;
            }
            count.emplace_back(cnt-1);
            Y2.to_device(q,count.data());
        }
        
    }

    ::std::tuple<at::Tensor, at::Tensor, at::Tensor> unique2(Tensor const& self,bool sorted,bool return_inverse, bool return_counts){
        GUARD;
        Tensor out0,out1,out2;
        if(self.numel() == 0){
            return std::tuple<Tensor,Tensor,Tensor>(out0, out1, out2); 
        }
        Tensor self_c = self.contiguous();
        dlprim::Tensor X = todp(self_c);
        self_c=self_c.reshape(X.shape().total_size());
        
        Tensor tmp1 = new_ocl_tensor(c10::IntArrayRef(X.shape().total_size()),self.device(),self.dtype().toScalarType());
        Tensor tmp2 = new_ocl_tensor(c10::IntArrayRef(X.shape().total_size()),self.device(),c10::kLong);
        auto out=std::get<0>(sort_values_stable(self_c,false,0,false,tmp1,tmp2));
        dlprim::Tensor Y = todp(out);
        dlprim::ExecutionContext q=getExecutionContext(self);
        Y.to_host(q);
        void *p = Y.host_data();
        X.to_host(q);
        void *ori = X.host_data();
        switch(Y.dtype()) {
        case dlprim::float_data:
            unique2impl(static_cast<float *>(p),static_cast<float *>(ori),X.shape().total_size(),self.device(),c10::kFloat,return_inverse,return_counts,out0,out1,out2,q);
            break;
        case dlprim::double_data:
            unique2impl(static_cast<double *>(p),static_cast<double *>(ori),X.shape().total_size(),self.device(),c10::kDouble,return_inverse,return_counts,out0,out1,out2,q);
            break;
        case dlprim::int8_data:
            unique2impl(static_cast<int8_t *>(p),static_cast<int8_t *>(ori),X.shape().total_size(),self.device(),c10::kChar,return_inverse,return_counts,out0,out1,out2,q);
            break;
        case dlprim::uint8_data:
            unique2impl(static_cast<uint8_t *>(p),static_cast<uint8_t *>(ori),X.shape().total_size(),self.device(),c10::kByte,return_inverse,return_counts,out0,out1,out2,q);
            break;
        case dlprim::int16_data:
            unique2impl(static_cast<int16_t *>(p),static_cast<int16_t *>(ori),X.shape().total_size(),self.device(),c10::kShort,return_inverse,return_counts,out0,out1,out2,q);
            break;
        case dlprim::int32_data:
            unique2impl(static_cast<int32_t *>(p),static_cast<int32_t *>(ori),X.shape().total_size(),self.device(),c10::kInt,return_inverse,return_counts,out0,out1,out2,q);
            break;
        case dlprim::int64_data:
            unique2impl(static_cast<int64_t *>(p),static_cast<int64_t *>(ori),X.shape().total_size(),self.device(),c10::kLong,return_inverse,return_counts,out0,out1,out2,q);
            break;
        default:
            TORCH_CHECK(!"Not implemented dtype","Not implemented");
        }
        sync_if_needed(self.device());
        return std::tuple<Tensor,Tensor,Tensor>(out0, out1, out2); 
    }
    
    Tensor flip_(Tensor const& self, IntArrayRef dims){
        GUARD;
        if(self.numel() == 0){
            return self;
        }
        Tensor self_c = self.contiguous();
        dlprim::Tensor X = todp(self);
        Tensor out = new_tensor_as(X.shape(),self);
        dlprim::Tensor Y=todp(out);
        
        auto q = getExecutionContext(self);
        dlprim::Context ctx(q);

        std::vector<long> DL;
        for(auto s:X.shape()){
            DL.emplace_back(s);
        }
        std::vector<int> Dims;
        for(auto const &v:dims){
            Dims.emplace_back(v);
        }

        dlprim::Tensor new_DL(ctx,dlprim::Shape(DL.size()),dlprim::int64_data);
        new_DL.to_device(q,DL.data());
        dlprim::Tensor new_Dims(ctx,dlprim::Shape(Dims.size()),dlprim::int32_data);
        new_Dims.to_device(q,Dims.data());

        dlprim::core::flip(X,Y,new_DL,new_Dims,q);
        sync_if_needed(self.device());
        return out;

    }

} // namespace dtype
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
      m.impl("aten::empty.memory_format", &ptdlprim::allocate_empty);
      m.impl("aten::empty_strided",&ptdlprim::empty_strided);
      m.impl("aten::_reshape_alias",&ptdlprim::_reshape_alias);
      m.impl("aten::view",&ptdlprim::view);
      m.impl("aten::_copy_from",&ptdlprim::_copy_from);
      m.impl("aten::fill_.Scalar",&ptdlprim::fill_);
      m.impl("aten::zero_",&ptdlprim::zero_);
      m.impl("aten::zeros",&ptdlprim::zeros);
      m.impl("aten::as_strided",&ptdlprim::as_strided);
      m.impl("aten::_local_scalar_dense",&ptdlprim::_local_scalar_dense);
      m.impl("aten::masked_select",&ptdlprim::masked_select);
      m.impl("aten::arange",&ptdlprim::arange);
      m.impl("aten::index.Tensor_out",&ptdlprim::index_Tensor_out);
      m.impl("aten::_index_put_impl_",&ptdlprim::index_put_impl);
      m.impl("aten::max.dim_max",&ptdlprim::max_dim_max);
      m.impl("aten::sort.values_stable",&ptdlprim::sort_values_stable);
      m.impl("aten::nonzero",&ptdlprim::nonzero);
      m.impl("aten::_unique2",&ptdlprim::unique2);
      m.impl("aten::flip",&ptdlprim::flip_);
}
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
      m.fallback(torch::CppFunction::makeFromBoxedFunction<&ptdlprim::fallback>());
}
