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
    namespace core { class LayerNormFwd; }

    struct LayerNormConfig {
        int features = -1;
        float eps = 1e-5f;
        bool affine = true;
        bool use_global_stats = false;
        static LayerNormConfig from_json(json::value const &v);
    };


    class LayerNorm : public Operator {
    public:
        
        LayerNorm(Context &ctx,LayerNormConfig const &config = LayerNormConfig(),DataType dtype=float_data);
        virtual ~LayerNorm();
        
        virtual char const *operator_type() const
        {
            return "LayerNorm";
        }
        
        virtual void initialize_params(std::vector<Tensor> &parameters,ExecutionContext const &e);
        virtual void mode(CalculationsMode m);
        virtual CalculationsMode mode() { return Operator::mode(); }

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           std::vector<TensorSpecs> &parameters,
                           size_t &workspace);

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out,
                             size_t &ws);

		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &output,
                             std::vector<Tensor> &parameters,
                             Tensor &workspace,
                             ExecutionContext const &ctx);


    private:

        
        Tensor combined_scale_,combined_bias_;
        LayerNormConfig config_;
        DataType dtype_;
        Shape setup_shape_;

        std::unique_ptr<core::LayerNormFwd> ln_gpu_;
    };
} // 
