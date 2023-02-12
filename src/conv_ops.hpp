#pragma once
#include "init.h"
#include "TensorOps.hpp"


namespace nn
{


	template<typename T, u32 N = 7>
	using Tensor = beta::Tensor<T, N>;
	template<typename T>
	using Vector = Tensor<T, 1>;

	template<typename T>
	Tensor<T> conv2d(
		const Tensor<T>& input,
		const Tensor<T>& kernel,
		const std::pair<i32, i32>& strides,
		const std::string& padding = "valid"
	)
	{
		auto conv_window = [](
			const Tensor<T>& x,
			const std::pair<size_t,size_t> k,
			const std::string& pad,
			const std::pair<size_t, size_t> stride,
			const bool ff = 1)
		{
			std::size_t
				n = x.shape()[0],
				h = x.shape()[1],
				w = x.shape()[2],
				c = x.shape()[3];
			auto [kh, kw] = k;
			auto [sh, sw] = stride;

			f64 h2, w2, ph, pw;
			if (pad == "valid")
			{
				ph = pw = 0;
				h2 = std::ceil(
					(f64)(h - kh + 1) / (f64)sh
				);
				w2 = std::ceil(
					(f64)(w - kw + 1) / (f64)sw
				);
				
			}
			else  
			{
				h2 = std::ceil((f64)h /(f64)sh);
				w2 = std::ceil((f64)w / (f64)sw);

				ph = std::max((f64)((h2 - 1) * sh + kh - h), (f64)0);
				pw = std::max((f64)((w2 - 1) * sw + kw - w), (f64)0);
			}
		
			i64 ph0 = std::floor((f64)ph/2.);
			i64 ph1 = std::ceil((f64)ph / 2.);
			i64 pw0 = std::floor((f64)pw / 2.);
			i64 pw1 = std::ceil((f64)pw / 2.);
			std::pair<size_t, size_t> pph, ppw;
			if (ff)
			{
				pph = { ph0,ph1 };
				ppw = { pw0,pw1 };
			}
			else {
				pph = { ph1,ph0 };
				ppw = { pw1,pw0 };
			}
			Tensor<T> padded = ops::pad(x,
				smallvec<std::pair<size_t,size_t>>{ {0, 0}, pph, ppw, { 0,0 } }) ;
			
			std::size_t
				x_sn = padded.strides()[0],
				x_sh = padded.strides()[1],
				x_sw = padded.strides()[2],
				x_sc = padded.strides()[3];

			smallvec<size_t> oshape{ (size_t)n,(size_t)h2, (size_t)w2, (size_t)kh, (size_t)kw, (size_t)c };
			smallvec<size_t> ostride{ x_sn, (size_t)(sh * x_sh), (size_t)(sw * x_sw),
				x_sh , x_sw , x_sc };

			return Tensor<T>(oshape, ostride, padded.offset(), padded.data());
		};

		Tensor<T> padded = conv_window(
			input, std::pair{ kernel.shape()[0] ,kernel.shape()[1]}
		, padding, strides);
		
		Tensor<T> k = kernel.reshape(
			smallvec<size_t>{kernel.shape()[0] * kernel.shape()[1] * kernel.shape()[2], kernel.shape()[3]}
		);
		smallvec<size_t> wshape = padded.shape();
		smallvec<size_t> reshape_x(2);
		reshape_x[0] = (size_t)(padded.shape()[0] * padded.shape()[1] * padded.shape()[2]);
		reshape_x[1] = padded.size() / reshape_x[0];

		padded = padded.reshape(reshape_x);
		
		Tensor<T> out = math::dot(padded, k);
		smallvec<size_t> s1 = { wshape[0], wshape[1],
			wshape[2] };
		s1.append(out.size() / s1.prod());
		return out.reshape(s1);

	}

	//computes the 2d convolution with strides(n,n) and VALID padding
	template<typename T>
	Tensor<T> conv2d(
		const Tensor<T>& input,
		const Tensor<T>& w,
		f64 stride = 1
	)
	{
		assert(input.rank() ==4 && w.rank() == 4 && "arguments have to be 4d tensors");
		auto Strides = [](const auto& vec) ->smallvec<size_t>
		{
			smallvec<size_t> outstride;
			for (i32 i = 0; i < 3; i++)
				outstride.append(vec[i]);
			for (i32 i = 1; i < vec.size(); i++)
				outstride.append(vec[i]);
			return outstride;
		};
		size_t hout =
			std::floor(
				(f64)(input.shape()[1] - w.shape()[0]) / stride + 1);
		size_t wout =
			std::floor(
				(f64)(input.shape()[2] - w.shape()[1]) / stride + 1);
		
		smallvec<size_t> _shape{ input.shape()[0],hout,wout,w.shape()[0],
			w.shape()[1],input.shape()[3] };
		smallvec<size_t> _strides = Strides(input.strides());
		_strides[1] *= stride;
		_strides[2] *= stride;

		Tensor<T> a(_shape, _strides, input.offset(), input.data());

		return math::tensordot(a, w, 3);
	}



};

namespace grad{

	template<typename T>
	Tensor<T> conv2d_dx(
		const Tensor<T>& out_grad,
		const Tensor<T>& kernel,
		const std::pair<size_t, size_t> resolution,
		const std::pair<i32, i32>& strides,
		const std::string& padding 
	)
	{
		auto getpad = [](
			const std::string& pad,
			const std::pair<size_t,size_t> res,
			const std::pair<i32, i32> strides,
			const std::pair<size_t, size_t> outres,
			const std::pair<size_t, size_t> kernel_size
		) -> std::pair<size_t,size_t>
		{
			auto [r1, r2] = res;
			auto [s1, s2] = strides;
			auto [or1, or2] = outres;
			auto [k1, k2] = kernel_size;
			if (pad == "valid")
				return { (k1 - 1) * 2 , (k2 - 1) * 2 };
			//else same
			f64 p1 = or1 + k1 - 1 - ( (r1 - 1) * s1 + 1 );
			f64 p2 = or2 + k2 - 1 - ( (r2 - 1) * s2 + 1 );
			return { std::clamp(p1, 0., (k1 - 1) * 2.)    ,std::clamp(p2, 0., (k2 - 1) * 2.) };
		};
		auto conv_window = [](
			const Tensor<T>& x,
			const std::pair<size_t, size_t> k,
			const std::pair<size_t, size_t> resolution,
			const std::pair<size_t , size_t> pad,
			const std::pair<size_t, size_t> strides,
			const bool ff = 1)
		{
			std::size_t
				n = x.shape()[0],
				h = x.shape()[1],
				w = x.shape()[2],
				c = x.shape()[3];
			auto [ph, pw] = pad;
			auto [kh, kw] = k;
			auto [sh, sw] = strides;
			auto [h2, w2] = resolution;

			Tensor<T> xx = tensor::zeros<T>(
				smallvec<size_t>{ n , h , sh , w , sw , c }
			);
			for (size_t i = 0; i < n; ++i)
				for (size_t j = 0; j < h; ++j)
					for (size_t ii = 0; ii < w; ++ii)
						for (size_t jj = 0; jj < c; ++jj)
							xx(i, j, 0, ii, 0, jj) = x(i, j, ii, jj);

			smallvec<size_t> bsh = xx.shape();

			xx = xx.reshape(bsh[0], bsh[1] * bsh[2],
				bsh[3] * bsh[4], bsh[5]);

			xx = ops::slice(xx, { {},{0, (i64)h2},{0, (i64)w2}/*,:*/});

			size_t ph2 = std::ceil(ph / 2.),
				ph3 = std::floor(ph / 2.),
				pw2 = std::ceil(pw / 2.),
				pw3 = std::floor(pw / 2.);
			std::pair<size_t, size_t> pph , ppw;
			if (ff) {
				pph = { ph3 ,  ph2 };
				ppw = { pw3 , pw2 };
			}
			else {
				pph = { ph2 , ph3 };
				ppw = { pw2 , pw3 };
			}

			Tensor<T> padded = ops::pad(xx,
				smallvec<std::pair<size_t, size_t>>{ {0,0},pph,ppw,{0,0} });

			std::size_t
				x_sn = padded.strides()[0],
				x_sh = padded.strides()[1],
				x_sw = padded.strides()[2],
				x_sc = padded.strides()[3];
			smallvec<size_t> ostr{
				x_sn , x_sh , x_sw , x_sh ,x_sw , x_sc
			};
			smallvec<size_t> osh{
				n ,h2 , w2 , kh , kw , c
			};
			return Tensor<T>(osh, ostr, padded.offset(), padded.data());
		};


		assert(kernel.shape()[-1] == out_grad.shape()[-1]);

		Tensor<T> wT = ops::transpose(kernel,
			{ 0,1,3,2 });

		std::pair<size_t, size_t>
			g12{ out_grad.shape()[1],out_grad.shape()[2] };
		std::pair<size_t, size_t>
			kernel_size{ kernel.shape()[0],kernel.shape()[1] };

		std::pair<f64, f64> pad = getpad(
			padding, resolution, strides, g12, kernel_size
		);

		Tensor<T> wrtx = conv_window(out_grad,
			kernel_size, resolution , pad, strides );

		size_t fdim = wrtx.shape()[0] * wrtx.shape()[1] * wrtx.shape()[2];

		smallvec<size_t> preshape = wrtx.shape();

		wrtx = wrtx.reshape( fdim , wrtx.size() / fdim );

		Tensor<T> flipped = ops::flip(wT, { 0,1 });
		
		fdim = flipped.shape()[0] * flipped.shape()[1] * flipped.shape()[2];

		flipped = flipped.reshape(fdim, flipped.size() / fdim);

		Tensor<T> out = math::dot(wrtx, flipped);

		fdim = preshape[0]* preshape[1]* preshape[2];

		return out.reshape(preshape[0], preshape[1], preshape[2], out.size() / fdim);
	};


	template<typename T>
	Tensor<T> conv2d_dw(
		const Tensor<T>& out_grad,
		const Tensor<T>& inputs,
		const std::pair<size_t, size_t> kernel_size,
		const std::pair<i32, i32>& strides,
		const std::string& padding
	)
	{

		auto conv_window = [](
			const Tensor<T>& x ,
			const std::pair<size_t,size_t>outsize,
			const std::pair<i32, i32> strides,
			const std::pair<size_t,size_t> kernel_size,
			const std::string& pad,
			const bool ff  = 1 ) 
		{
			std::size_t
				n = x.shape()[0],
				h = x.shape()[1],
				w = x.shape()[2],
				c = x.shape()[3];
			auto [kh, kw] = outsize;
			auto [sh, sw] = strides;
			auto [h2, w2] = kernel_size;

			f64 ph, pw;
			if (pad == "valid")
				ph = pw = 0;
			else {
				ph = std::max(
					(f64)(h2 - 1) + ((kh - 1) * sh + 1) - h, 0.);
				pw = std::max(
					(f64)(w2 - 1) + ((kw - 1) * sw + 1) - w, 0.);
			}
			std::size_t
				ph2 = std::ceil(ph / 2.),
				ph3 = std::floor(ph / 2.),
				pw2 = std::ceil(pw / 2.),
				pw3 = std::floor(pw / 2.);

			std::pair<size_t, size_t> pph, ppw;
			if (ff)
			{
				pph = { ph3 , ph2 };
				ppw = { pw3 , pw2 };
			}
			else {
				pph = { ph2 , ph3 };
				ppw = { pw2 ,pw3 };
			}

			Tensor<T> padded = ops::pad(x,
				smallvec< std::pair<size_t, size_t>>
			{ {0, 0}, { ph3,ph2 }, { pw3 ,pw2 }, { 0,0 }});

			i64 p2h = (-(i64)padded.shape()[1]) % sh,
			p2w = (-(i64)padded.shape()[2]) % sw;
			if(p2h>0 || p2w >0)
				padded = ops::pad(
					padded ,
					smallvec< std::pair<size_t, size_t>>
			        { {0, 0}, { 0,ph2 }, { 0 ,pw2 }, { 0,0 }});

			std::size_t
				x_sn = padded.strides()[0],
				x_sh = padded.strides()[1],
				x_sw = padded.strides()[2],
				x_sc = padded.strides()[3];

			smallvec<size_t>ostr
			{ x_sn,x_sh,x_sw,sh * x_sh,sw * x_sw,x_sc };
			smallvec<size_t> osh
			{ n , h2 , w2 , kh , kw , c };

			return Tensor<T>(osh, ostr, padded.offset(), padded.data());
		};

		Tensor<T> gT = ops::transpose(out_grad, { 1,2,0,3 });
		Tensor<T> xT = ops::transpose(inputs, { 3,1,2,0 });

		std::pair<size_t, size_t> outsize{ gT.shape()[0] , gT.shape()[1] };

		xT = conv_window(xT, outsize,  strides, kernel_size, padding);

		smallvec<size_t> bsh = xT.shape();

		gT = gT.reshape(
			gT.shape()[0] * gT.shape()[1] * gT.shape()[2],
			gT.shape()[3]
		);

		size_t fdim = bsh[0] * bsh[1] * bsh[2];

		xT = xT.reshape(fdim, xT.size() / fdim);

		Tensor<T> out = math::dot(xT, gT);

		out = out.reshape(bsh[0],
			bsh[1], bsh[2], out.size() / fdim);

		out = ops::transpose(out, { 1,2,0,3 });

		out = ops::slice(out,
			{ {0 , (i64)kernel_size.first},{0, (i64)kernel_size.second}/*,: ,:*/ });
		return out;
	}
};