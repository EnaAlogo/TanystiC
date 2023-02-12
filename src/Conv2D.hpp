#pragma once
#include "Layer.hpp"
#include "UniquePtr.hpp"
#include "nn_ops.hpp"
#include "initializers.hpp"
#include "backwards.hpp"
#include "conv_ops.hpp"

template<typename T>
class Conv2D : public Layer<T>
{
	using Tensor = beta::Tensor<T, 7>;
	using Matrix = beta::Tensor<T, 2>;
	using Vector = beta::Tensor<T, 1>;

public:
	using value_type = Layer<T>::value_type;
	
	Conv2D(
		const size_t filters,
		const std::pair<size_t,size_t> kernel_size,
		const std::pair<i32,i32> strides={1,1},
		const std::string& padding ="valid" ,
		const bool use_bias = true ,
		T(*activation)(T) = nullptr,
		const std::string& kernel_initializer = "uniform",
		const std::string& bias_initializer = "zeros"
	)
		:
		filters(filters),
		kernel_size(kernel_size),
		pad(padding),
		strides(strides),
		use_bias(use_bias),
		activation(activation),
		kernel_initializer(initializers::get<T>(kernel_initializer)),
		bias_initializer(initializers::get<T>(bias_initializer))
	{};

	Tensor call(const Tensor& inputs, const bool training) override
	{
		if (!built)
			build(inputs);

		if (training)
			stored_inputs = inputs;

		Tensor output = nn::conv2d(inputs, kernel, strides, pad);

		if (use_bias)
			nn::bias_add(output, bias);

		if (activation) {
			if(training)
			    stored_outputs = output.deepcopy();
			output.apply(activation);
		}

		return output;
	}
	Tensor backwards(const Tensor& out_grad, f64 lr) override
	{
		std::pair<size_t, size_t> image_resolution
		{ stored_inputs.shape()[1],
		stored_inputs.shape()[2] };

		Tensor grad = out_grad;
		if (activation){
			stored_outputs.apply([this](T item)
				{return grad::derivative(activation, item);}
			);
			ops::Multiply(grad, stored_outputs, stored_outputs);
			grad = stored_outputs;
		}
		Tensor dx = grad::conv2d_dx(grad, kernel, image_resolution, strides, pad);

		Tensor dw = grad::conv2d_dw(grad, stored_inputs, kernel_size, strides, pad);

		dw *= lr;

		nn::rm(kernel, dw);

		if (use_bias){
			grad *= lr;
			nn::rm(bias, grad);
		}

		return dx;
	}
	void build(const Tensor& inputs) override
	{
		if (inputs.rank() != 4)
			std::cerr << "conv2d excepts 4d tensor", throw std::invalid_argument("");
		auto [I, J] = kernel_size;

		kernel = Tensor({ I,J,inputs.shape()[-1] , filters });

		kernel.initialize(*kernel_initializer);

		if (use_bias)
		{
			bias = Vector({ filters });
			bias.initialize(*bias_initializer);
		}
		else {
			if (bias_initializer)
				bias_initializer.release();
		}
		built = true;
	}
private:
	Tensor kernel;
	Tensor stored_inputs;
	Tensor stored_outputs;
	Vector bias;
	const std::pair<size_t, size_t> kernel_size;
	const std::pair<i32, i32> strides;
	UniquePtr<initializer<T>> kernel_initializer;
	UniquePtr<initializer<T>> bias_initializer;
	const std::string pad;
	const size_t filters;
	T(*activation)(T);
	const bool use_bias;
	bool built = false;
};