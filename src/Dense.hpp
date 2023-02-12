#pragma once
#include "init.h"
#include "Tensor.hpp"
#include "UniquePtr.hpp"
#include "Layer.hpp"
#include "TensorOps.hpp"
#include "nn_ops.hpp"
#include "backwards.hpp"

template<typename _type,typename =
std::enable_if_t<std::is_floating_point_v<_type>>>
class Dense : public Layer<_type>
{
 
	
	using Tensor = beta::Tensor<_type, 7>;

	using Matrix = beta::Tensor<_type, 2>;

	using Vector = beta::Tensor<_type, 1>;
public:
	using value_type = _type;


	Dense(
		const size_t units,
		value_type(*activation)(value_type) = nullptr,
		const bool usebias = true,
		const std::string& kernel_initializer = "uniform",
		const std::string& bias_initializer = "zeros")
		: 
		units(units), activation(activation),
		kernel_initializer(initializers::get<value_type>(kernel_initializer)),
		bias_initializer(initializers::get<value_type>(bias_initializer)),
		use_bias(usebias) {};


	
	Tensor call(const Tensor& inputs, bool training ) override
	{
		if (!built)
			build(inputs);

		stored_inputs = inputs;
		
		Tensor outputs = math::tensordot(inputs, weights, std::array{ -1 ,0 });

		if (use_bias)
			nn::bias_add(outputs, bias);
		if (activation) {
			if(training)
			   stored_outputs = outputs.deepcopy();
			outputs.apply(activation);
		}
		return outputs;
	}


	Tensor backwards(const Tensor& out_grad, f64 lr) override
	{
		Tensor grad = out_grad;
		if (activation)
		{
			stored_outputs.apply(
				[this](value_type item)
				{
					return grad::derivative(activation, item);
				}
			);
			ops::Multiply(grad, stored_outputs, grad);
		}
		Tensor dx = math::dot(grad, weights.T());

		Tensor dw = math::dot(stored_inputs.T(), grad);

		dw *= lr;
		nn::rm(weights, dw);

		if (use_bias) {
			Tensor db = grad;
			db *= lr;
			nn::rm(bias, db);
		}
		return dx;
	}


	void build(const Tensor& inputs) override
	{
		
		weights = Matrix({ inputs.shape()[-1] , units });
		weights.initialize( *kernel_initializer );
		if (use_bias)
		{
			bias = Vector({ units });
			bias.initialize( *bias_initializer );
		}
		else {
			if (bias_initializer)
				bias_initializer.release();
		}
		built = true;
	}

	Matrix& weight()
	{
		return weights;
	}
	Vector& biases()
	{
		return bias;
	}
private:
	Matrix weights;
	Vector bias;
	Tensor stored_inputs;
	Tensor stored_outputs;
	UniquePtr<initializer<value_type>> kernel_initializer;
	UniquePtr<initializer<value_type>> bias_initializer;
	value_type(*activation)(value_type);
	const size_t units;
	bool use_bias;
	bool built = false;


};