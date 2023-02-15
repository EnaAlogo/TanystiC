#pragma once
#include "init.h"
#include "Layer.hpp"
#include "nn_ops.hpp"
#include "TensorOps.hpp"
#include "backwards.hpp"

template<typename T>
class ReLU : public Layer<T>
{
	using Tensor = Layer<T>::Tensor;
public:
	using value_type = T;


	Tensor call(const Tensor& input, const bool training) override
	{
		Tensor y = nn::relu(input);
		if (training)
			stored_activation = y;
		return y;
	}
	Tensor backwards(const Tensor& out_grad, f64 lr) override
	{
		stored_activation.apply([](T item) { return item > 0 ? 1 : 0; });
		ops::Multiply(stored_activation, out_grad, stored_activation);
		return stored_activation;
	}

private:
	Tensor stored_activation;
};

template<typename T>
class ELU : public Layer<T>
{
	using Tensor = Layer<T>::Tensor;
public:
	using value_type = T;

	ELU(const T alpha = 1)
		:alpha(alpha) {};

	Tensor call(const Tensor& input, const bool training) override
	{
		
		Tensor y = input.transform([this](T z)
			{
				return z < 0 ? alpha * (std::exp(z) - 1) : z;
			});
		if (training)
			stored_activation = y;
		return y;
	}

	Tensor backwards(const Tensor& out_grad, f64 lr) override
	{
		//delta elu return 1 if z >0 else alpha * exp(z)
		// a( ( a(exp(z)-1) )/ a +1 ) = a*exp(z) maybe
		T a1 = 1 / alpha;
		stored_activation.apply(
			[this , a1](T z)
			{
				return z > 0 ? 1 : (z * a1 + 1) * alpha;
			}
		);
		ops::Multiply(stored_activation, out_grad, stored_activation);
		return stored_activation;

	}

private:
	Tensor stored_activation;
	const T alpha;
};

template<typename T>
class LeakyReLU : public Layer<T>
{
	using Tensor = Layer<T>::Tensor;
public:
	using value_type = T;

	LeakyReLU(const T alpha = .3)
		:alpha(alpha) {};

	Tensor call(const Tensor& input, const bool training) override
	{

		Tensor y = input.transform([this](T z)
			{
				return  z >= 0 ? z : alpha * z;
			});
		if (training)
			stored_activation = y;
		return y;
	}

	Tensor backwards(const Tensor& out_grad, f64 lr) override
	{
		stored_activation.apply(
			[this](T z)
			{
				return z > 0 ? 1 : alpha;
			}
		);
		ops::Multiply(stored_activation, out_grad, stored_activation);
		return stored_activation;

	}

private:
	Tensor stored_activation;
	const T alpha;
};

template<typename T>
class GELU : public Layer<T>
{
	using Tensor = Layer<T>::Tensor;
public:


	Tensor call(const Tensor& input, const bool training) override
	{
		auto invsq2 = 1 / std::numbers::sqrt2_v<T>;
		if (training) {
			inputs = input.deepcopy();
			cdf = input.transform(
				[invsq2](T x)
				{
					return .5 * (1 + std::erf(x * invsq2));
				}
			);
			return ops::Multiply(cdf, input);
		}
		else
			return input.transform([invsq2](T x)
				{
					return x * .5 * (1 + std::erf(x * invsq2));
				});
	}


	Tensor backwards(const Tensor& out_grad, f64 lr) override
	{
		auto insvqp = (1 / std::numbers::sqrt2_v<T>) * std::numbers::inv_sqrtpi_v<T>;
		Tensor xexp2p = inputs.apply(
			[insvqp](T x)
			{
				return x * std::exp(-((x * x) * .5)) * insvqp;
			}
		);
		ops::Add(cdf, xexp2p, cdf);
		ops::Multiply(out_grad, cdf, cdf);
		return cdf;
	}

private:
	Tensor cdf;
	Tensor inputs;
};


template<typename T>
class Sigmoid : public Layer<T>
{
	using Tensor = Layer<T>::Tensor;
public:
	using value_type = T;

	Tensor call(const Tensor& input, const bool training) override
	{
		Tensor y = nn::sigmoid(input);
		if (training)
			stored_activation = y;
		return y;
	}

	Tensor backwards(const Tensor& out_grad, f64 lr) override
	{
		stored_activation.apply(
			[](T item)
			{
				return item * (1 - item);
			}
		);
		ops::Multiply(stored_activation, out_grad, stored_activation);
		return stored_activation;

	}

private:
	Tensor stored_activation;
};



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


private:
	Tensor stored_activation;
};




template<typename T>
class Softmax : public Layer<T>
{
	using Tensor = Layer<T>::Tensor;
public:
	using value_type = T;

	Softmax(const i32 axis = -1)
		:axis(axis) {};

	Tensor call(const Tensor& input, const bool training) override
	{

		T max = input.max();
		Tensor exp = input.transform(
			[&max = std::as_const(max)]
		(T x) {return std::exp(x - max); });


		ops::Divde(exp, reduce::sum(exp, { axis }, 1), exp);

		if (training)
			stored_activation = exp;

		return  exp;
	}

	Tensor backwards(const Tensor& out_grad, f64 lr) override
	{

		return ops::Multiply(out_grad, grad::softmaxdx(stored_activation));
	}

private:
	const i32 axis;
	Tensor stored_activation;


};


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
		ops::Multiply(out_grad, df, df);
		return df;
	}


private:
	std::function<Tensor(const Tensor&)> func;
	Tensor stored_input;
};