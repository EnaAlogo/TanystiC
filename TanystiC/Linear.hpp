#pragma once
#include "init.h"
#include "Tensor.hpp"
#include "UniquePtr.hpp"
#include "TensorOps.hpp"
#include "nn_ops.hpp"
#include "Layer.hpp"

template<typename _type, typename =
	std::enable_if_t<std::is_floating_point_v<_type>>>
class Linear : public Layer<_type>
{

	
	using Tensor = beta::Tensor<_type, 7>;

	using Matrix = beta::Tensor<_type, 2>;

	using Vector = beta::Tensor<_type, 1>;
public:
	using value_type = _type;


	Linear(
		const size_t units,
		const bool usebias = true,
		const std::string& kernel_initializer = "uniform",
		const std::string& bias_initializer = "zeros")
		:
		units(units),
		kernel_initializer(initializers::get<value_type>(kernel_initializer)),
		bias_initializer(initializers::get<value_type>(bias_initializer)),
		use_bias(usebias) {};

	
	Tensor call(const Tensor& inputs, const bool training ) override
	{
		if (!built)
			build(inputs);

		if(training)
		  stored_inputs = inputs;

		Tensor outputs = math::tensordot(inputs, weights, std::array{ -1 ,0 });

		if (use_bias)
			nn::bias_add(outputs, bias);

		return outputs;
	}

	Tensor backwards(const Tensor& out_grad , f64 lr) override
	{
		
		Tensor dx = math::dot(out_grad , weights.T());

		smallvec<i32> axes = vec::tovec<i32>(range(out_grad.rank() - 1));

		Tensor dw = math::tensordot(stored_inputs, out_grad, std::pair{ axes,axes });
		dw *= lr;
		nn::rm(weights, dw);

		if (use_bias) {
			Tensor db = reduce::sum(out_grad, { 0 });
			db *= lr;
			nn::rm(bias, db);
		}

		return dx;
	}

	void build(const Tensor& inputs) 
	{

		weights = Matrix({ inputs.shape()[-1] , units });
		weights.initialize(*kernel_initializer);
		if (use_bias)
		{
			bias = Vector({ units });
			bias.initialize(*bias_initializer);
		}
#if 0
		else {
			if (bias_initializer)
				bias_initializer.release();
		}
#endif
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
	UniquePtr<initializer<value_type>> kernel_initializer;
	UniquePtr<initializer<value_type>> bias_initializer;
	const size_t units;
	const bool use_bias;
	bool built = false;

	
};