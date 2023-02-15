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
		const T eps = 1e-5,
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
		Tensor sigma = var.transform([this](T x) {return std::sqrt(x + eps); });
		Tensor xcenter = ops::Subtract(inputs, mean);
		Tensor xnorm = ops::Divde(xcenter, sigma);
		if (training) {
			std = sigma;
			x_centered = xcenter;
			x_norm = xnorm;
		}
		
		smallvec<size_t> bcastshape(inputs.rank());
		for (size_t& i : bcastshape) i = 1;
		bcastshape[-1] = inputs.shape()[-1];

		Tensor scale_ = ops::broadcast_to(gamma, bcastshape);
		Tensor offset = ops::broadcast_to(beta, bcastshape);

		Tensor out = xnorm.deepcopy();

		ops::Divde(xnorm, sigma, out);

		if (scale)
			ops::Multiply(scale_, out, out);

		if (center)
			ops::Add(out, offset, out);
		return out;

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

		built = 1;
	}

	Tensor backwards(const Tensor& out_grad, f64 lr)
	{
		Tensor dx_norm = scale ? ops::Multiply(out_grad, gamma) : out_grad;
		T n = gamma.size();
			/*[&out_grad]() {
			f64 prod=1;
			for (i32 i = 1; i < out_grad.rank() ; ++i)
				prod *= out_grad.shape()[i];
			return prod;
		}(); */
		T invn = 1 / n;
		std.apply([&invn](T x) {return invn / x; });
		Tensor dx = ops::Multiply(
			ops::Subtract(
				ops::Subtract(dx_norm * n, reduce::sum(dx_norm, { -1 } , 1)),
				ops::Multiply(x_norm, reduce::sum(ops::Multiply(dx_norm, x_norm), { -1 } ,1))
			),
			std );

		if (scale)
			nn::rm(gamma *= lr, reduce::sum(ops::Multiply(out_grad, x_norm), {0} ));
		if (center)
			nn::rm(beta *= lr, reduce::sum(out_grad ,{0} ));
		return dx;

	}

private:
	Tensor
		std,
		x_norm,
		x_centered;
	Vector
		beta,
		gamma;
	UniquePtr<initializer<T>>
		beta_initializer,
		gamma_initializer;
	const T eps;
	const bool center, scale;
	bool built = false;
};