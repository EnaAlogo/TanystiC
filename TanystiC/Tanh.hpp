#pragma once
#include "Layer.hpp"
#include "nn_ops.hpp"

template<typename T>
class Tanh : public Layer<T>
{
	using Tensor = Layer<T>::Tensor;
public:
	using value_type = T;

	Tensor call(const Tensor& input, const bool training) override
	{
		Tensor y = nn::tanh(input);
		if (training)
			stored_activation = y;
		return y;
	}
	Tensor backwards(const Tensor& out_grad, f64 lr) override
	{
		stored_activation.apply(
			[](T item)
			{
				return 1 - (item * item);
			});
		ops::Multiply(stored_activation, out_grad, stored_activation);
		return stored_activation;
	}

	void build(const Tensor& input) override {};

private:
	Tensor stored_activation;
};