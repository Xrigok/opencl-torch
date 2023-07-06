///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2023-2024 Jinpo Xu <xu675217572@gmail.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/operator.hpp>
namespace dlprim {	
    namespace json { class value; }
    namespace core { class BMMForward; }

    struct BatchMatMulConfig {
        int M1 = 0;
        int M2 = 0;
        int K = 0;
        bool BatchMatMulConfig = false;
        static BatchMatMulConfig from_json(json::value const &v);
    };


    class BatchMatMul : public Operator {
    public:
        
        BatchMatMul(Context &ctx,BatchMatMulConfig const &config = BatchMatMulConfig(),DataType dtype=float_data);
        virtual ~BatchMatMul();
        
        virtual char const *operator_type() const
        {
            return "BatchMatMul";
        }
        

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> const &weight,
                           std::vector<TensorSpecs> &out,
                           size_t &workspace);

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out,
                             size_t &ws);

		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &weight,
                             std::vector<Tensor> &result,
                             Tensor &workspace,
                             ExecutionContext const &ctx);
        
        virtual void get_output_shape(Shape const &in,Shape const &weight);


    private:

        BatchMatMulConfig config_;
        DataType dtype_;
        Shape setup_shape_;

        std::unique_ptr<core::BMMForward> bmm_gpu_;
    };
} // 
