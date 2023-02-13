#pragma once
#include "Layer.hpp"
#include "normalization.hpp"
#include "initializers.hpp"

template<typename T>
class LayerNorm : public Layer<T>
{
	using Tensor = Layer<T>::Tensor;
	using Vector = Layer<T>::Vector;
public:
LayerNorm(
		const f64 eps = 1e-5,
		const bool center = true,
		const bool scale = true,
		const std::string& beta_initializer = "zeros",
		const std::string& gamma_initializer = "ones"

	)
		:eps(eps), scale(scale), center(center),
		beta_initializer(initializers::get<T>(beta_initializer)),
		gamma_initializer(initializers::get<T>(gamma_initializer))
	{};

	Tensor call(const Tensor& inputs, const bool training) override
	{
		if (!built)
			build(inputs);

		Tensor mean = reduce::mean(inputs, { -1 } , 1);
		Tensor var = reduce::variance(inputs, { -1 } , 1);
		if (training) {
			std = reduce::stddev(inputs, { -1 },1);
			x_norm = ops::Divde(ops::Subtract(inputs, mean), std);
		}

		smallvec<size_t> bcastshape(inputs.rank());
		for (size_t& i : bcastshape) i = 1;
		bcastshape[-1] = inputs.shape()[-1];

		Tensor scale = ops::broadcast_to(gamma, bcastshape);
		Tensor offset = ops::broadcast_to(beta, bcastshape);

		return nn::batch_norm(inputs, mean, var, scale, offset, eps);

	}

	void build(const Tensor& input) override
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

		built = 1;
	}

	Tensor backwards(const Tensor& out_grad, f64 lr)
	{

		auto [dx, db, dg] = grad::batch_norm_backwards(out_grad, x_norm, std, gamma);

		nn::rm(gamma *= lr, dg);
		nn::rm(beta *= lr, db);

		return dx;

	}

private:
	Tensor
		std,
		x_norm;
	Vector
		beta,
		gamma;
	UniquePtr<initializer<T>>
		beta_initializer,
		gamma_initializer;
	const f64 eps;
	const bool center, scale;
	bool built = false;
};