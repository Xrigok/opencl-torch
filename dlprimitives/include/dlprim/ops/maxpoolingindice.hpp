///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/operator.hpp>

using namespace torch;

using c10::Device;
using c10::DeviceType;

namespace dlprim {	
    namespace json { class value; }

    struct MaxPoolingIndiceMulConfig {
        int kernel[2] = {1,1};
		int stride[2] = {1,1};
		int pad[2] = {0,0};
        int dilation[2] = {1,1};
        static MaxPoolingIndiceMulConfig from_json(json::value const &v);
    };


   
    class MaxPoolingIndice : public Operator {
    public:
        
        MaxPoolingIndice(Context &ctx,MaxPoolingIndiceMulConfig config = MaxPoolingIndiceMulConfig());
        virtual ~MaxPoolingIndice();
        
        virtual char const *operator_type() const
        {
            return "Max_Pooling_Indice";
        }

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &indices,
                           std::vector<TensorSpecs> &parameters,
                           size_t &workspace);

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out,
                             std::vector<Shape> &indices,
                             size_t &ws);

		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &indices,
                             std::vector<Tensor> &parameters,
                             Tensor &workspace,
                             ExecutionContext const &ctx);
        
    private:

        DataType dtype_;
    };
} // namespace
 
