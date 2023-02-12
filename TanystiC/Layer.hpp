#pragma once
#include "init.h"

template<typename T>
class Layer
{
protected:
	using Tensor = beta::Tensor<T, 7>;
	using Vector = beta::Tensor<T, 1>;
public:
	using value_type = T;

	virtual Tensor call(const Tensor& input,const bool training)  = 0;

	virtual Tensor backwards(const Tensor& grad, f64 lr) = 0;

	virtual void build(const Tensor& input) = 0;

	Tensor operator()(const Tensor& input , bool training = 0)
	{
		return call(input, training);
	}

	virtual ~Layer() {};
};