#pragma once
#include "Layer.hpp"
#include "normalization.hpp"
#include "initializers.hpp"

template<typename T>
class BatchNorm : public Layer<T>
{
	using Tensor = Layer<T>::Tensor;
	using Vector = Layer<T>::Vector;
public:

	BatchNorm(
		const T momentum = .99,
		const T eps = 1e-5,
		const bool center = true,
		const bool scale = true,
		const std::string& beta_initializer = "zeros",
		const std::string& gamma_initializer = "ones",
		const std::string& moving_mean_initializer = "zeros",
		const std::string& moving_var_initializer="ones"
		
	)
		:momentum(momentum) ,eps(eps),scale(scale),center(center),
		beta_initializer(initializers::get<T>(beta_initializer)),
		gamma_initializer(initializers::get<T>(gamma_initializer)),
		moving_mean_initializer(initializers::get<T>(moving_mean_initializer)),
		moving_var_initializer(initializers::get<T>(moving_var_initializer))
	{};

	Tensor call(const Tensor& inputs, const bool training) override
	{
		if (!built)
			build(inputs);

		if (training) {
			Tensor in_mean = reduce::mean(inputs, { 0 });
			Tensor in_var = reduce::variance(inputs, { 0 });

			moving_mean *= momentum;
			moving_var *= momentum;
			ops::Add(moving_mean, in_mean.transform([this](T x)
				{
					return (1 - momentum)* x;
				}),moving_mean);

			ops::Add(moving_var, in_var.transform([this](T x)
				{
					return (1 - momentum) * x;
				}), moving_var);

			std = in_var.transform([this](T x)
				{
					return std::sqrt(x + eps);
				});

			x_centered = ops::Subtract(inputs, in_mean);
			x_norm = ops::Divde(x_centered, std);

			Tensor out = x_norm.deepcopy();

			if(scale)
				ops::Multiply(gamma , x_norm , out);
			if (center)
				nn::bias_add(out, beta);

			return out;
		}
		else {
			Vector epsq = moving_var.transform(
				[this](T item) {return std::sqrt(item + eps); }
			);
			Tensor x_norm = ops::Subtract(inputs, moving_mean);

			Tensor out = x_norm.deepcopy();
			

			ops::Divde(x_norm, epsq, out);

			if (scale)
				ops::Multiply(gamma, x_norm, out);

			if (center)
				nn::bias_add(out, beta);
			return out;
		}
	}

	void build(const Tensor& input) 
	{
		size_t units = input.shape()[-1];

		if (center) {
			beta = Vector({ units });
			beta.initialize(*beta_initializer);
		}
		if (scale) {
			gamma = Vector({ units });
			gamma.initialize(*gamma_initializer);
		}

		moving_mean = Vector({ units }),
			moving_var = Vector({ units });
		moving_mean.initialize(*moving_mean_initializer);
		moving_var.initialize(*moving_var_initializer);

		built = 1;
	}

	Tensor backwards(const Tensor& out_grad, f64 lr)
	{

		Tensor dx_norm = scale ? ops::Multiply(out_grad, gamma) : out_grad;
		T n = [&out_grad]() {
			f64 prod=1;
			for (i32 i = 0; i < out_grad.rank() - 1; ++i)
				prod *= out_grad.shape()[i];
			return prod;
		}();
		T invn = 1 / n;
		std.apply([&invn](T x) {return invn / x; });
		Tensor dx = ops::Multiply(
			ops::Subtract(
				ops::Subtract(dx_norm * n, reduce::sum(dx_norm, { 0 })),
				ops::Multiply(x_norm, reduce::sum(ops::Multiply(dx_norm, x_norm), { 0 }))
			) , 
			std );

		if (scale)
			nn::rm(gamma *= lr, reduce::sum(ops::Multiply(out_grad, x_norm), { 0 }));
		if (center)
			nn::rm(beta *=  lr, reduce::sum(out_grad , {0}));
		return dx;
	}

private:
	Tensor
		std,
		x_norm,
		x_centered;
	Vector 
		moving_mean ,
		moving_var,
		beta,
		gamma;
	UniquePtr<initializer<T>> 
		beta_initializer,
		gamma_initializer, moving_mean_initializer,
		moving_var_initializer;
	const T
		momentum, 
		eps;
	const bool center, scale;
	bool built = false;
};