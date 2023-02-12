#pragma once
#include "Layer.hpp"
#include "nn_ops.hpp"
#include "TensorOps.hpp"

template<typename T>
class Softmax : public Layer<T>
{
	using Tensor = Layer<T>::Tensor;
public:
	using value_type = T;

	Tensor call(const Tensor& input, const bool training) override
	{
		Tensor y = nn::stable_softmax(input);
		if (training)
			stored_activation = y;
		return y;
	}

	Tensor backwards(const Tensor& out_grad, f64 lr) override
	{
		return { 0 };
	}

	void build(const Tensor& input) override {};

private:
	Tensor stored_activation;

	Tensor vectorizeY()
	{
		return ops::squeeze(stored_activation).
			reshape(smallvec<size_t, 2>{stored_activation.size(), 1});
	}
};