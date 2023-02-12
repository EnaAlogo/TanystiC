#pragma once
#include "Layer.hpp"
#include "backwards.hpp"

template<typename T>
class Lambda : public Layer<T>
{
	using Tensor = beta::Tensor<T, 7>;

public:
	using value_type = T;

	Lambda(const std::function<Tensor(const Tensor&)>& func)
		:func(func) {};

	Tensor call(const Tensor& input, const bool training) override
	{
		if (training)
			stored_input = input;
		return func(input);
	}
	Tensor backwards(const Tensor& out_grad, f64 lr) override
	{
		Tensor df = grad::gradient(func, stored_input);
		return df;
	}
	void build(const Tensor& input) override
	{};


private:
	std::function<Tensor(const Tensor&)> func;
	Tensor stored_input;
};