#pragma once
#include "init.h"
#include "Layer.hpp"

template<typename T>
class GELU : public Layer<T>
{
	using Tensor = Layer<T>::Tensor;
public:


	Tensor call(const Tensor& input,const bool training) override
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

	void build(const Tensor& input) override  {};

	Tensor backwards(const Tensor& out_grad, f64 lr) override
	{
		auto insvqp = (1 / std::numbers::sqrt2_v<T>)*std::numbers::inv_sqrtpi_v<T>;
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