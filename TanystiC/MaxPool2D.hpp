#pragma once
#include "Layer.hpp"
#include "pooling.hpp"
#include "nn_ops.hpp"

template<typename T>
class MaxPool2D : public Layer<T>
{
	using Tensor = Layer<T>::Tensor;

public:
	MaxPool2D(
		const std::pair<i32, i32>pool_size,
		const std::pair<i32, i32> strides = { 1,1 }
	) :pool_size(pool_size), strides(strides) {};

	Tensor call(const Tensor& inputs, const bool training) override {
		if (training)
			this->inputs = inputs;
		return nn::max_pool2d(inputs, pool_size, strides);
	}


	Tensor backwards(const Tensor& out_grad, f64 lr) override
	{
		
		return grad::Maxpool2d_backwards(out_grad, inputs, pool_size, strides);
	}

private:
	const std::pair<i32, i32> pool_size;
	const std::pair<i32, i32> strides;
	Tensor inputs;
};