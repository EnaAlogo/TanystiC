#pragma once
#include "init.h"
#include "TensorOps.hpp"
#include "nn_ops.hpp"
#include "Tensorfuncs.hpp"


namespace nn
{
	template<typename T ,u32 N, u32 M ,u32 K>
	Tensor<T, N> batch_norm(
		const Tensor<T,N>& x,
		const Tensor<T, K>& rmean,
		const Tensor<T, K>& rvar,
		const Tensor<T, M>& gamma,
		const Tensor<T, M>&beta,
		f64 eps=1e-5
	)
	{
		Tensor<T, N> epsq = rvar.transform(
			[eps](T item) {return std::sqrt(item + eps); }
		);
		Tensor<T, N> x_norm = ops::Subtract(x, rmean);
		ops::Multiply(gamma, x_norm , x_norm);
		ops::Divde(x_norm, epsq , x_norm);
		ops::Add(x_norm, beta, x_norm);
		return x_norm;
	}
	
};

namespace grad
{

	template<typename T, u32 N , u32 M, u32 K>
	std::array<Tensor<T, N>,3> batch_norm_backwards(
		const Tensor<T,N> out_grad,
		const Tensor<T, N>& x_norm,
		const Tensor<T, K>& std,
		const Tensor<T, M>& gamma
	)
	{
		Tensor<T, N> dg = reduce::sum(
			ops::Multiply(out_grad, x_norm), { 0 });
		Tensor<T, N> db = reduce::sum(out_grad, { 0 });

		Tensor<T, N> dxn = ops::Multiply(out_grad, gamma);

		T n = std.shape().prod();

		Tensor<T, N> dxsum = reduce::sum(dxn, { 0 });

		Tensor<T, N> dxxsum = 
			ops::Multiply(x_norm,reduce::sum(ops::Multiply(dxn, x_norm), { 0 }));
		T invN = 1. / n;
		Tensor<T, N> invstd = std.transform(
			[invN](T item) {return invN / (item + 1e-5); }
		);
		Tensor<T, N> dx = ops::Multiply(
			invstd, ops::Subtract(ops::Subtract(dxn *= n, dxsum), dxxsum)
		);

		return { dx , db , dg };
	}

	template<typename T, u32 N, u32 M, u32 K>
	std::array<Tensor<T, N>, 3> layer_norm_backwards(
		const Tensor<T, N> out_grad,
		const Tensor<T, N>& x_norm,
		const Tensor<T, K>& std,
		const Tensor<T, M>& gamma
	)
	{
		Tensor<T, N> dg = reduce::sum(
			ops::Multiply(out_grad, x_norm), { -1 });
		Tensor<T, N> db = reduce::sum(out_grad, { -1 });

		Tensor<T, N> dxn = ops::Multiply(out_grad, gamma);

		T n = std.shape().prod();

		Tensor<T, N> dxsum = reduce::sum(dxn, { -1 });

		Tensor<T, N> dxxsum =
			ops::Multiply(x_norm, reduce::sum(ops::Multiply(dxn, x_norm), { -1 }));
		T invN = 1. / n;
		Tensor<T, N> invstd = std.transform(
			[invN](T item) {return invN / (item + 1e-5); }
		);
		Tensor<T, N> dx = ops::Multiply(
			invstd, ops::Subtract(ops::Subtract(dxn *= n, dxsum), dxxsum)
		);

		return { dx , db , dg };
	}

};