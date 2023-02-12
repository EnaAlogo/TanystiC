#pragma once
#include "Layer.hpp"
#include "backwards.hpp"
#include "TensorOps.hpp"

template<typename T>
class Residual :public Layer<T>
{
	using Tensor = beta::Tensor<T>;
public:
	using value_type = Layer<T>::value_type;

	Residual(Layer<T>* layer)
		:callableLayer(layer) {};

	Tensor call(const Tensor& input, const bool training) override
	{
		Tensor output = callableLayer->call(input , training);

		ops::Add(input, output, output);
		
		return output;
	}

	Tensor backwards(const Tensor& out_grad, f64 lr) override
	{
		return callableLayer->backwards(out_grad, lr);
	}

	void build(const Tensor& input) override
	{};

private:
	Layer<T>* callableLayer;

};