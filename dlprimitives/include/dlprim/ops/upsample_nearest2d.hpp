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
    namespace core { class UpsampleNearest2dFwd; }

    struct UpsampleNearest2dConfig {
        double s_h=1;
        double s_w=1;
        static UpsampleNearest2dConfig from_json(json::value const &v);
    };


    class UpsampleNearest2d : public Operator {
    public:
        
        UpsampleNearest2d(Context &ctx,UpsampleNearest2dConfig const &config = UpsampleNearest2dConfig(),DataType dtype=float_data);
        virtual ~UpsampleNearest2d();
        
        virtual char const *operator_type() const
        {
            return "upsample_nearest2d";
        }
        

		virtual void setup(std::vector<TensorSpecs> const &in,
                           std::vector<TensorSpecs> &out,
                           size_t &workspace);

        virtual void reshape(std::vector<Shape> const &in,
                             std::vector<Shape> &out,
                             size_t &ws);

		virtual void forward(std::vector<Tensor> &input,
                             std::vector<Tensor> &result,
                             Tensor &workspace,
                             ExecutionContext const &ctx);
        
        virtual void get_output_shape(Shape const &in,Shape const &weight);


    private:

        UpsampleNearest2dConfig config_;
        DataType dtype_;
        Shape setup_shape_;

        std::unique_ptr<core::UpsampleNearest2dFwd> up_gpu_;
    };
} // 
