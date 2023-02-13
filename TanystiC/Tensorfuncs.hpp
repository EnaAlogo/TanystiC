#pragma once
#include "init.h"
#include "smallvec.hpp"
#include "TensorOps.hpp"

#include "TensorIterators.hpp"




namespace cum {

	template<typename T, u32 N>
	beta::Tensor<T, N > sum(const beta::Tensor<T, N>& a, i32 axis)
	{

		beta::Tensor<T, N> result(a.shape());
		assert(axis < a.rank());

		axis = axis >= 0 ? axis : axis + a.rank();
		for (auto [index, item] : tEnumerate(result))
		{
			size_t i = a.calc_offset(index);
			item = a[i];
			if (index[axis] != 0)
				item += result[i - result.strides()[axis]];
		}

		return result;
	}
	template<typename T, u32 N>
	beta::Tensor<T, N > multiply(const beta::Tensor<T, N>& a, i32 axis)
	{

		beta::Tensor<T, N> result(a.shape());
		assert(axis < a.rank());

		axis = axis >= 0 ? axis : axis + a.rank();
		for (auto [index, item] : tEnumerate(result))
		{
			size_t i = a.calc_offset(index);
			item = a[i];
			if (index[axis] != 0)
				item *= result[i - result.strides()[axis]];
		}

		return result;
	}
	template<typename T, u32 N>
	beta::Tensor<T, N > mean(const beta::Tensor<T, N>& a, i32 axis)
	{

		// Compute the sum of the elements along the specified axis
		beta::Tensor<T, N > sum = cum::sum(a, axis);
		// Compute the size of the reduction dimension
		size_t dim_size = a.shape()[axis];
		// Divide the sum by the size of the reduction dimension to get the mean
		sum /= dim_size;
		return sum;
	}

	

};

namespace reduce {
	template<typename T, u32 N>
	using Tensor = beta::Tensor < T, N>;

	template<typename T>
	using Vector = beta::Tensor<T, 1>;


	namespace _internal {

		template<typename T, u32 N>
		std::tuple<beta::Tensor<T, N>, beta::Tensor<T, N>, smallvec<size_t, N>>
			set_up_reduction(
				const beta::Tensor<T, N>& a,
				std::initializer_list<i32> axes,
				const T initial_value,
				const bool keepdims = false
			)
		{
			assert(axes.size() <= a.rank());
			smallvec<size_t, N>new_shape;
			smallvec<size_t, N> new_str;
			smallvec<i32, N> axes_;
			for (i32 i : axes)
				axes_.append(i >= 0 ? i : i + a.rank());

			for (i32 i : axes) {
				new_shape.append(a.shape()[i]);
				new_str.append(a.strides()[i]);
			}
			for (i32 i = 0; i < a.rank(); i++)
				if (!axes_.contains(i)) {//todo keepdims
					new_shape.append(a.shape()[i]);
					new_str.append(a.strides()[i]);
				}
			beta::Tensor<T, N> result(vec::slice(new_shape, axes.size(), a.rank()));
			for (auto& item : result)
				item = initial_value;

			auto strides = ops::_internal::bcast_strides(result, new_shape);
			return { beta::Tensor<T,N>(new_shape , new_str,
									   a.offset() , a.data())  ,result , strides };
		}
	};
	template<typename T, u32 N>
	beta::Tensor<T, N> sum(const beta::Tensor<T, N>& a, std::initializer_list<i32> axes,
		const bool keepdims = 0 )
	{

		auto [tmp, out, strides] = _internal::set_up_reduction(a, axes, (T)0);

		for (auto [index, item] : tEnumerate(tmp))
		{
			size_t i = vec::compute_flat_index(strides, tmp.shape(), index);
			out[i] += item;
		}
		if (keepdims) {
			smallvec<size_t, N> keepdims = a.shape();
			for (auto i : axes) 
				keepdims[i] = 1;
			out = out.reshape(keepdims);
		}
		return out;
	}
	template<typename T, u32 N>
	T sum(const beta::Tensor<T, N>& a)
	{
		T s = 0;
		for (const auto& item : a)
			s += item;
		return s;
	}
	template<typename T, u32 N>
	beta::Tensor<T, N> multiply(const beta::Tensor<T, N>& a, std::initializer_list<i32> axes,
		const bool keepdims = 0)
	{
		auto [tmp, out, strides] = _internal::set_up_reduction(a, axes, (T)1);
		for (auto [index, item] : tEnumerate(tmp))
		{
			size_t i = vec::compute_flat_index(strides, tmp.shape(), index);
			out[i] *= item;
		}
		if (keepdims) {
			smallvec<size_t, N> keepdims = a.shape();
			for (auto i : axes)
				keepdims[i] = 1;
			out = out.reshape(keepdims);
		}
		return out;
	}
	template<typename T, u32 N>
	T multiply(const beta::Tensor<T, N>& a)
	{
		T s = 1;
		for (const auto& item : a)
			s *= item;
		return s;
	}
	template<typename T, u32 N>
	beta::Tensor<T, N> mean(const beta::Tensor<T, N>& a, std::initializer_list<i32> axes,
		const bool keepdims =0 )
	{
		// Compute the sum of the elements along the specified axis
		auto total = sum(a, axes , keepdims);

		//TODO mean sht
		// Compute the size of the reduction dimension
		T prod = 1;
		for (auto i : axes)
			prod *= a.shape()[i];

		// Divide the sum by the size of the reduction dimension to get the mean
		total /= prod;
		return total;
	}
	template<typename T, u32 N>
	T mean(const beta::Tensor<T, N>& a)
	{
		// Compute the sum of the elements along the specified axis
		auto total = sum(a);
		size_t dim_size = a.size();
		// Divide the sum by the size of the reduction dimension to get the mean
		total /= dim_size;
		return total;
	}

	template<typename T, u32 N>
	Tensor<T, N> variance(const Tensor<T, N>& a, std::initializer_list<i32> axes,
		const bool keepdims=0)
	{
		T prod = 1;
		for (auto i : axes) {
			prod *= a.shape()[i];
		};

		Tensor<T,N> mean_ = mean(a , axes , 1);


		Tensor<T, N> x = ops::Subtract(a, mean_ );

		x.apply([](T item) {return item * item; });

		Tensor<T, N> ret = sum(x, axes , keepdims);

		ret /= prod;

		return ret;

	}

	template<typename T, u32 N>
	T variance(const Tensor<T, N>& a)
	{
		T mean_ = mean(a);
		Tensor<T, N> m = a - mean_;
		m.apply([](T item) {return item * item; });
		T sum_ = sum(m);
		size_t n = a.size();
		return sum_ / n;
	};

	template<typename T, u32 N>
	T stddev(const Tensor<T, N>& a)
	{
		T var = variance(a);
		return std::sqrt(var);
	};
	template<typename T, u32 N>
	Tensor<T, N> stddev(const Tensor<T, N>& a, std::initializer_list<i32> axes,
		const bool keepdims = 0)
	{
		Tensor<T, N> var = variance(a, axes , keepdims);
		var.apply([](T item) {return std::sqrt(item); });
		return var;
	};

	template<typename T, u32 N>
	Tensor<T, N> norm(const Tensor<T, N>& a, std::initializer_list<i32> axes,
		const bool keepdims =0)
	{
		
		auto [tmp, out, strides] = _internal::set_up_reduction(a, axes, (T)0);

		for (auto [index, item] : tEnumerate(tmp))
		{
			size_t i = vec::compute_flat_index(strides, tmp.shape(), index);
			out[i] += item*item;
		}
		out.apply([](T item) {return std::sqrt(item); });
		if (keepdims) {
			smallvec<size_t, N> keepdims = a.shape();
			for (auto i : axes)
				keepdims[i] = 1;
			out = out.reshape(keepdims);
		}
		return out;
	}

	template<typename T,  u32 N > 
	T norm(const Tensor<T, N>& a)
	{
		T norm_ = 0;
		for (const auto& item : a)
			norm_ += item * item;
		return std::sqrt(norm_);
	}
#if 0 
	template<typename T , u32  N>
	Tensor<T, N> max(const Tensor<T, N>& a, std::initializer_list<i32> axes)
	{
		
	}
#endif
};