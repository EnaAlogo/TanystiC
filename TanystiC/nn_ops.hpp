#pragma once
#include "init.h"
#include "TensorOps.hpp"
#include "TensorIterators.hpp"
#include "Tensorfuncs.hpp"

namespace nn
{

	template<typename T,u32 N = 7>
	using Tensor = beta::Tensor<T, N>;
	template<typename T >
	using Vector = beta::Tensor<T, 1>;
	template<typename T >
	using Matrix = beta::Tensor<T, 2>;
#if 0 
	template<typename T , u32 N>
	std::pair<Tensor<T, N>, Tensor<T, N>> moments(const Tensor<T, N>& x)
	{

	};
#endif
	template<typename T , u32 N>
	Tensor<T>& bias_add(Tensor<T , N>& output, const Vector<T>& biases)
	{
		smallvec<size_t , N > bcast_strides = ops::_internal::bcast_strides(
			biases, output.shape()
		);
		for (auto [i, item] : tEnumerate(output))
		{
			size_t bias_index = vec::compute_flat_index(bcast_strides, output.shape(), i);
			item += biases[bias_index];
		}
		return output;
	}

	template<typename T , u32 N , u32 K >
	void rm(Tensor<T, N>& a, const Tensor<T, K>& b)
	{
		if (b.size() == 1)
			a -= b[0];
		else
			ops::Subtract(a, b, a);
	}

	template<typename T , u32 N>
	Tensor<T> linear(
		const Tensor<T , N>& inputs,
		const Matrix<T>& weight,
		const std::optional<Vector<T>>& bias = std::nullopt)
	{
		
		auto outputs = math::tensordot(inputs, weight, std::array{ -1 , 0 });

		if (bias.has_value())
			bias_add(outputs, bias.value());

		return outputs;
	}

	

	



	template<typename T >
	Tensor<T> softmax(const Tensor<T>& logits)
	{
		Tensor<T> inexp = math::exp(logits);
		return inexp /= reduce::sum(inexp);
	}
	template<typename T >
	Tensor<T> stable_softmax(const Tensor<T>& logits)
	{
		T max = logits.max();
		Tensor<T> y = logits.transform(
			[max](T item)
			{
				return std::exp(item - max);
			}
		);
		return y /= reduce::sum(y);
	}
	template<typename T>
	T cross_entropy(const Tensor<T>& input, const Tensor<T>& target)
	{
		const auto eps = 1e-12;//std::numeric_limits<T>::epsilon();
		assert(input.shape()[0] == target.shape()[0]);
		T loss = 0;
		T m = (T)target.shape()[0];
		for (auto [y_pred, y] : zip(input, target))
			loss += y * std::log(y_pred + eps);
		return  -loss / m;
	}
	template<typename T>
	T sparse_categorical_cross_entropy(const Tensor<T>& input, const Tensor<T>& target)
	{
		const auto eps = 1e-12;//std::numeric_limits<T>::epsilon();
		assert(input.shape()[0] == target.shape()[0]);
		T loss = 0;
		T m = (T)target.shape()[0];
		for (size_t i = 0; i < m; ++i)
			for (size_t j = 0; j < input.shape()[1]; ++j)
				loss += (target(i, 0)==j) * std::log(input(i, j) + eps);
		
		return  -loss / m;
	}
	template<typename T >
	T binary_cross_entropy(const Tensor<T>& input, const Tensor<T>& Target)
	{
		const auto eps = 1e-12;//std::numeric_limits<T>::epsilon();
		assert(input.shape()[0] == Target.shape()[0]);
		T loss = 0;
		T m = (T)Target.shape()[0];
		for (auto [y_pred, y] : zip(input, Target))
			loss += (y* std::log(y_pred + eps)) + ((1 - y) * std::log(1 - y_pred + eps));
		return -loss / m;

	}

	template<typename T>
	T mean_squared_error(const Tensor<T>& input, const Tensor<T>& Target)
	{
		Tensor<T> mse = ops::Subtract(Target , input);
		mse.apply([](T item) {return item * item; });
		return reduce::mean(mse);
	}

	template<typename T , u32 N>
	Tensor<T, N> relu(const Tensor<T, N>& input)
	{
		return input.transform([](T item) {return std::max((T)0, item); });
	}
	template<typename T , std::enable_if_t<std::is_arithmetic_v<T>>>
	T relu(T el)
	{
		return std::max(el, (T)0);
	};
	template<typename T , u32 N>
	Tensor<T, N> sigmoid(const Tensor<T, N>& input)
	{
		return input.transform([](T item) {return 1 / (1 + std::exp(-item)); });
	}
	template<typename T>
	T sigmoid(T el)
	{
		return 1 / (1 + std::exp(-el));
	}
	template<typename T, u32 N>
	Tensor<T, N> tanh(const Tensor<T, N>& input)
	{
		return input.transform([](T item) {return std::tanh(item); });
	}



	
};


namespace activations
{
	template<typename T , u32 N = 7>
	using Tensor = beta::Tensor<T, N>;

	namespace _internal {
		template<typename ...Act>
		struct aggregate : Act...
		{
			using Act::operator()...;
		};
	};

	template<typename T , u32 N>
	void relu(Tensor<T, N>& input)
	{
		input.apply([](T item) {return std::max(0, item); });
	}

	template<typename T, u32 N>
	void tanh(Tensor<T, N>& input)
	{
		input.apply([](T item) {return std::tanh(item); });
	}
	template<typename T, u32 N>
	void elu(Tensor<T, N>& input, f64 alpha = 1.)
	{
		//delta elu return 1 if z >0 else alpha * exp(z)
		input.apply([alpha](T z)
			{
				if (z < 0)
				  z = alpha * (std::exp(z) - 1);
			});
	}
	template<typename T, u32 N>
	void leaky_relu(Tensor<T, N>& input)
	{
		input.apply([](T z)
			{
				   return  std::max(.1 * z, z);
			});

	}

	template<typename T, u32 N>
	void gelu(Tensor<T, N>& input)
	{
		input.apply([](T z)
			{
				return .5 * z * (1 + std::tanh(
					(std::numbers::inv_pi_v<T> * 2) *
					(z + .044715 * (z * z * z) ) ) );
			});

	}

	template<typename T , u32 N>
	void sigmoid(Tensor<T, N>& input)
	{
		input.apply([](T item) {return 1 / (1 + std::exp(-item)); });
	}

	template<typename T, u32 N>
	void softmax(Tensor<T, N>& logits, const bool use_shift = 1)
	{
		if (use_shift)
		{
			T max = logits.max();
			logits.apply([&max](T item) {return std::exp(item - max); });
		}
		else
			logits.apply([](T item) {return std::exp(item); });

		T sum = reduce::sum(logits);
		logits /= sum;
	}
};
