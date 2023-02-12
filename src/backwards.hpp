#pragma once
#include "init.h"
#include "nn_ops.hpp"
#include "TensorBase.h"
#include "TensorOps.hpp"
#include "Tensorfuncs.hpp"

namespace nn
{
	namespace training
	{

		template<typename T ,u32 M, u32 N>
		auto step(
			f32 lr ,
			const beta::Tensor<T,M>& input ,
			Matrix<f32>& weight,
			Vector<f32>& bias, 
			Tensor<T, N>& grad_output)
		{
			auto grad_input = math::tensordot(grad_output , weight.T(),std::array{-1,0});

			auto grad_weights = math::tensordot(input.T(), weight, std::array{ -1,0 });

			auto grad_biases = reduce::mean(grad_output) * input.shape()[0];

			grad_weights *= lr;
			grad_biases *= lr;

			ops::Subtract(weight, grad_weights, weight);
			bias -= grad_biases;

			return grad_input;

		}
		template<typename T, u32 M>
		auto step(
			f32 lr,
			const beta::Tensor<T, M>& input,
			Matrix<f32>& weight,
			Vector<f32>& bias,
			T grad_output)
		{
			auto grad_input = weight.T() * grad_output;
			
			auto grad_weights = weight * grad_output;

			auto grad_biases = grad_output * input.shape()[0];

			grad_weights *= lr;
			grad_biases *= lr;

			ops::Subtract(weight, grad_weights, weight);
			bias -= grad_biases;

			return grad_input;
		}


	};
	
	
	
};

namespace grad
{
	template<typename T, u32 N = 7>
	using Tensor = beta::Tensor<T , N>;
	template<typename T>
	using Matrix = beta::Tensor<T, 2>;
	template<typename T>
	using Vector = beta::Tensor<T, 1>;

	static constexpr auto e_ = 1e-6;
	static constexpr auto e_sqr = 1e-3;

	template<typename T , u32 N>
	Tensor<T,N> gradient(
		const auto& F,
		const Tensor<T, N>& x)
	{
		Tensor<T, N> h(x.shape());
		/*
		* create the H tensor for diff to hold the h vals{x*sqrt(eps)} 
		* or a constant , when x == 0 to avoid division by 0
		*/ 
		for (auto [i, item] : enumerate(x))
			h[i] = item == 0 ? e_ : item * e_sqr;
		// the first term F(x+h)
		Tensor<T, N> xph = ops::Add(x, h);
		Tensor<T, N> fxph = F(xph);
		/* 
		* subtract 2h from the x + h tensor to create the x - h tensor 
		* without copying, 2h will be used for division (F(x+h)-F(x-h)) / 2h
		*/
		ops::Subtract(xph, h *= 2, xph);
		// create the second term F(x-h)
		Tensor<T, N> fxsh = F(xph);
		/*
		* take the difference from the first and the second term 
		* without making a new tensor/copying
		*/ 
		ops::Subtract(fxph, fxsh, fxph);
		//finally divide (F(x+h)-F(x-h)) / 2h without copying anything
		ops::Divde(fxph, h, fxph);

		return fxph;
	}

	template<typename T, typename func,
		typename = std::enable_if_t<std::is_arithmetic_v<T>>>
	inline T derivative(const func& F, T x)
	{
		const auto h = !x ? e_ : x * e_sqr;

		return (F(x + h) - F(x - h)) / (2 * h);
	}

	template<typename T>
	Tensor<T> grad_bce(const Tensor<T>& input, const Tensor<T>& Target)
	{
		auto negt = -Target;
		auto negp = -input;
		return ops::Add(
			ops::Divde(negt + 1, negp + 1),
			ops::Divde(negt, input)
		) /= Target.shape()[0];
	}
	template<typename T>
	Tensor<T> delta_bce(const Tensor<T>& input, const Tensor<T>& target)
	{
		Tensor<T> deltace{ input.shape() };
		for (size_t i = 0; i < input.shape()[0]; ++i)
		  for (size_t j = 0; j < input.shape()[-1]; ++j){
			  deltace(i, j) = ((1 - target(i, 0)) / (1 - input(i, j))
				  - target(i, 0) / input(i, j))
				  /target.shape()[0];

		  }
		return deltace;
	}
	template<typename T>
	Tensor<T> grad_mse(const Tensor<T>& input, const Tensor<T>& Target)
	{
		return (ops::Subtract(input, Target) *= 2) /= Target.shape()[0];
	}

	template<typename T>
	Tensor<T> delta_sparse_ce(const Tensor<T>& input, const Tensor<T>& Target)
	{


		size_t m = Target.shape()[0];
		size_t k = input.shape()[-1];
		Tensor<T> dL({ m,k });
		for (size_t i = 0; i < m; ++i)
				for (size_t y = 0; y < k; ++y)
					dL(i, y) = input(i, y) - (Target(i, 0)==y);

		return dL;
	}

	template<typename T>
	Tensor<T> delta_ce(const Tensor<T>& input, const Tensor<T>& Target)
	{
		return ops::Subtract(input, Target);
	}

	template<typename T>
	Tensor<T> grad_softmax(const Tensor<T>& input, const Tensor<T>& softmax_ouput)
	{
		size_t n = softmax_ouput.size();
		auto res = ops::Subtract(tensor::identity<T>(n), softmax_ouput.T());
		ops::Multiply(res, softmax_ouput, res);
		return res;
	}

	template<typename T>
	Tensor<T> softmaxdx(const Tensor<T>& s)
	{
		Tensor<T> dx(s.shape());
		auto sgrad = [](const Tensor<T>& s, Tensor<T>& dx, smallvec<size_t>& indices)
		{
			for (size_t i = 0; i < s.shape()[-2]; ++i) {
				indices[-2] = i;
				for (size_t j = 0; j < s.shape()[-1]; ++j) {
					indices[-1] = j;
					size_t dxi = dx.calc_offset(indices);
					if (i == j) {
						indices[-1] = i;
						size_t si = s.calc_offset(indices);
						dx[dxi] = s[si] * (1 - s[si]);
					}
					else {
						size_t si = s.calc_offset(indices);
						dx[dxi] = -s[si] * s[si];
					}
						
				}
			}
		};

		smallvec<size_t> i(s.rank());
		linalg::_internal::const_matrixloop(s, dx, i,0, sgrad);
		return dx;
	}

	template<typename T>
	Tensor<T> delta_softmax(const Tensor<T>& softmax_ouput)
	{
		size_t b = softmax_ouput.shape()[0];
		size_t n = softmax_ouput.shape()[-1];
		Tensor<T> jacobian{ b,n ,n};
		for (size_t u = 0; u < b; u++)
		  for (size_t i = 0; i < n; ++i)
			for (size_t j = 0; j < n; ++j)
			  if (i == j)
				jacobian(u, i, j) = softmax_ouput(u, i) * (1 - softmax_ouput(u, i));
			  else
				jacobian(u, i, j) = -softmax_ouput(u, i) * softmax_ouput(u, j);
		return jacobian;
	}

	template<typename T>
	Tensor<T> g_softmax(const Tensor<T>& softmax_ouput)
	{

		Tensor<T> outer = -math::outer(softmax_ouput, softmax_ouput);
		return ops::Add(outer, ops::diag(softmax_ouput.reshape(
			smallvec<size_t>{softmax_ouput.size()})));

#if 0
		Tensor<T> eye = tensor::identity<T,7>(softmax_ouput.shape()[-1]);
		ops::Subtract(eye, softmax_ouput.T(), eye);
		ops::Multiply(softmax_ouput, eye , eye);
	return eye; // assumes no batching to fix use identity tensor of (B , n , n )
#endif
	}

	template<typename T, u32 N>
	std::array<Tensor<T>, 2> linear_backwards(
		const Tensor<T>& out_grad,
		const Tensor<T, N>& inputs,
		const Matrix<T>& weight
	)
	{
		auto dx = math::dot(out_grad, weight.T());
		
		auto dw = math::dot(inputs.T(), out_grad);

		return { dx , dw };
	}

	template<typename t>
	Tensor<t> grad_simgoid(const Tensor<t>& x)
	{
		return x.transform([](t i)
			{
				t sig = nn::sigmoid(i);
		        return sig * (1 - sig);
			});
	}
	template<typename t>
	Tensor<t> grad_tanh(const Tensor<t>& x)
	{
		return x.transform([](t i)
			{
				t tan = std::tanh(i);
		        return 1 - (tan * tan);
			});
	}
	template<typename t , std::enable_if_t<std::is_arithmetic_v<t>>>
	t derivative_sigmoid(t x)
	{
		t sig = nn::sigmoid(x);
		return sig * (1 - sig);
	}
	template<typename t , std::enable_if_t<std::is_arithmetic_v<t>>>
	t derivative_tanh(t x)
	{
		t tan = std::tanh(x);
		return 1 - (tan * tan);
	}
};