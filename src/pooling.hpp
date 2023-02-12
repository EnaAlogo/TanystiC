#pragma once
#include "init.h"
#include "TensorOps.hpp"
#include "nn_ops.hpp"
#include "backwards.hpp"
#include "Tensorfuncs.hpp"

namespace nn
{

	template<typename T , u32 N>
	Tensor<T, N> pool2d(
		const Tensor<T,N>&a,
		std::pair<i32 , i32 > size,
		std::pair<i32,i32> stride,
		const std::string& mode = "max"
	)
	{
		auto poolfunc =
			[](const std::string& mode)
		{
			if (mode == "max")
				return std::mem_fn(&Tensor<T,N>::max);
			else if (mode == "avg")
				return std::mem_fn(&Tensor<T, N>::mean);
			else
				std::cerr << "invalid pool mode", throw std::invalid_argument("");
		}(mode);
	
		std::size_t
			m = a.shape()[0], nh = a.shape()[1], nw = a.shape()[2], nc = a.shape()[3];
		nh = std::floor(1 + (nh - size.first) / (f64)stride.first),
			nw = std::floor(1 + (nw - size.second) / (f64)stride.second);

		Tensor<T, N> A({ m,nh,nw,nc });

		for (i64 i = 0; i < m; ++i) {
		  for (i64 h = 0; h < nh; ++h) {
			i64 vs = h * stride.first,
				  ve = vs + size.first;
			for (i64 w = 0; w < nw; ++w) {
			  i64	hs = w * stride.second,
					he = hs + size.second;
			  for (i64 c = 0; c < nc; ++c) {
				
				Tensor<T, N> pslice = ops::slice(a, { {i,i + 1},{vs,ve},{hs,he},{c,c + 1} });

				A(i, h, w, c) = poolfunc(pslice);

			  }
			}
		  }
		}

		return A;

	}

	template<typename T, u32 N>
	Tensor<T, N> max_pool2d(
		const Tensor<T, N>& a,
		std::pair<i32, i32 > size,
		std::pair<i32, i32> stride
	)
	{

		std::size_t
			m = a.shape()[0], nh = a.shape()[1], nw = a.shape()[2], nc = a.shape()[3];
		nh = std::floor(1 + (nh - size.first) / (f64)stride.first),
			nw = std::floor(1 + (nw - size.second) / (f64)stride.second);

		Tensor<T, N> A({ m,nh,nw,nc });

		for (i64 i = 0; i < m; ++i) {
		  for (i64 h = 0; h < nh; ++h) {
			  i64 vs = h * stride.first,
				  ve = vs + size.first;
		    for (i64 w = 0; w < nw; ++w) {
			  i64 hs = w * stride.second,
					he = hs + size.second;
		   	  for (i64 c = 0; c < nc; ++c) {

		   		Tensor<T, N> pslice = ops::slice(a, { {i,i + 1},{vs,ve},{hs,he},{c,c + 1} });
		   
		   		A(i, h, w, c) = pslice.max();
		   
		   	  }
		    }
		  }
		}

		return A;

	}

	template<typename T, u32 N>
	Tensor<T, N> avg_pool2d(
		const Tensor<T, N>& a,
		std::pair<i32, i32 > size,
		std::pair<i32, i32> stride
	)
	{

		std::size_t
			m = a.shape()[0], nh = a.shape()[1], nw = a.shape()[2], nc = a.shape()[3];
		nh = std::floor(1 + (nh - size.first) / (f64)stride.first),
			nw = std::floor(1 + (nw - size.second) / (f64)stride.second);

		Tensor<T, N> A({ m,nh,nw,nc });

		for (i64 i = 0; i < m; ++i) {
		  for (i64 h = 0; h < nh; ++h) {
			i64 vs = h * stride.first,
				ve = vs + size.first;
			for (i64 w = 0; w < nw; ++w) {
			  i64 hs = w * stride.second,
					he = hs + size.second;
			  for (i64 c = 0; c < nc; ++c) {
		      
		      	Tensor<T, N> pslice = ops::slice(a, { {i,i + 1},{vs,ve},{hs,he},{c,c + 1} });
		      
		      	A(i, h, w, c) = pslice.mean();
		      
		      }
		    }
		  }
		}

		return A;

	}
};


namespace grad
{
	namespace _internal {

		template<typename T , u32 N>
		void Maxpooling_step(
			Tensor<T, N>& dx, 
			const Tensor<T, N>& da,
			const Tensor<T,N>& input,
			const std::pair<i32,i32> _ ,
			i64 i ,i64 h , i64 w,i64 c,
			i64 vs,i64 ve ,i64 hs ,i64 he)
		{
			auto get_masked = [i,h,w,c](const Tensor<T,N>& da , const Tensor<T,N>& slice) {
				T max = slice.max();
				T mul = da(i, h, w, c);
				Tensor<T, N>out(slice.shape());
				for (const auto [i, item] : enumerate(slice))
					out[i] = item == max ? mul : (T)0;
				return out;
			};
			Tensor<T, N> aslice = ops::slice(input, { {i,i + 1},{vs,ve},{hs,he},{c,c + 1} });
			Tensor<T, N> window = ops::slice(dx, { {i,i + 1},{vs,ve},{hs,he},{c,c + 1} });
			ops::Add(window, get_masked(da,aslice), window);
		}

		template<typename T, u32 N>
		void Avgpooling_step(
			Tensor<T, N>& dx,
			const Tensor<T, N>& da,
			const Tensor<T, N>& _,
			const std::pair<i32,i32> size,
			i64 i, i64 h, i64 w, i64 c,
			i64 vs, i64 ve, i64 hs, i64 he)
		{
			auto undoavg = [](T da, const std::pair<i32, i32> shape)
			{
				auto [nh, nw] = shape;
				T average = da / static_cast<f64>((nh * nw));
				Matrix<T> res(smallvec<size_t,2>{ static_cast<size_t>(nh),static_cast<size_t>(nw) });
				for (auto& el : res)
					el = average;
				return res;
			};
			T d = da(i, h, w, c);
			Tensor<T, N> window = ops::slice(dx, { {i,i + 1},{vs,ve},{hs,he},{c,c + 1} });
			ops::Add(window, undoavg(d, size), window);
		}
	};

	template<typename T, u32 N>
	Tensor<T, N> pool2d_backwards(
		const Tensor<T, N>& out_grad,
		const Tensor<T,N>& input,
		std::pair<i32, i32 > size,
		std::pair<i32, i32> stride,
		const std::string& mode = "max"
	)
	{
		Tensor<T, N> dx = tensor::zeros<T>(input.shape());
		auto poolfunc = [&out_grad, &input , &size](Tensor<T>& ref,const std::string& mode)
		{
			if (mode == "max")
				return std::bind_front(&_internal::Maxpooling_step<T,N>,
					ref, out_grad, input , size);
			else if (mode == "avg")
				return std::bind_front(&_internal::Avgpooling_step<T, N>,
					ref, out_grad, input , size);
			else
				std::cerr << "invalid pool mode", throw std::invalid_argument("");
		}(dx , mode);

		std::size_t m = out_grad.shape()[0], nh = out_grad.shape()[1],
			nw = out_grad.shape()[2], nc = out_grad.shape()[3];

		for (i64 i = 0; i < m; ++i) {
		  for (i64 h = 0; h < nh; ++h) {
			i64 vs = h * stride.first,
					ve = vs + size.first;
			for (i64 w = 0; w < nw; ++w) {
			  i64 hs = w * stride.second,
						he = hs + size.second;
			  for (i64 c = 0; c < nc; ++c) {
		  	  	poolfunc(i, h, w, c, vs, ve, hs, he);
		  	  }
		  	}
		  }
		}
		return dx;
	}

	template<typename T, u32 N>
	Tensor<T, N> Maxpool2d_backwards(
		const Tensor<T, N>& out_grad,
		const Tensor<T, N>& input,
		std::pair<i32, i32 > size,
		std::pair<i32, i32> stride
	)
	{
		Tensor<T, N> dx = tensor::zeros<T>(input.shape());

		std::size_t m = out_grad.shape()[0], nh = out_grad.shape()[1],
			nw = out_grad.shape()[2], nc = out_grad.shape()[3];

		for (i64 i = 0; i < m; ++i) {
		  for (i64 h = 0; h < nh; ++h) {
		  	i64 vs = h * stride.first,
		  		ve = vs + size.first;
		    for (i64 w = 0; w < nw; ++w) {
		      i64 hs = w * stride.second,
		      	he = hs + size.second;
		      for (i64 c = 0; c < nc; ++c) {
		   	  	_internal::Maxpooling_step(dx , out_grad, input , size,i, h, w, c, vs, ve, hs, he);
		   	  }
		   	}
		  }
		}
		return dx;
	}

	template<typename T, u32 N>
	Tensor<T, N> Avgpool2d_backwards(
		const Tensor<T, N>& out_grad,
		const Tensor<T, N>& input,
		std::pair<i32, i32 > size,
		std::pair<i32, i32> stride
	)
	{
		Tensor<T, N> dx = tensor::zeros<T>(input.shape());

		std::size_t m = out_grad.shape()[0], nh = out_grad.shape()[1],
			nw = out_grad.shape()[2], nc = out_grad.shape()[3];

		for (i64 i = 0; i < m; ++i) {
		  for (i64 h = 0; h < nh; ++h) {
		    i64 vs = h * stride.first,
		    	ve = vs + size.first;
		    for (i64 w = 0; w < nw; ++w) {
		      i64 hs = w * stride.second,
		      	he = hs + size.second;
		      for (i64 c = 0; c < nc; ++c) {
		      	_internal::Avgpooling_step(dx, out_grad, input, size, i, h, w, c, vs, ve, hs, he);
		      }
		    }
		  }
		}
		return dx;
	}
};