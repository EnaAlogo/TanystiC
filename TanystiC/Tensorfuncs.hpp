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

	namespace _internal{

		template<typename T, u32 N >
		std::tuple<Tensor<T, N>, Tensor<T, N>, Tensor<T, N>>
		_reduce(
				const Tensor<T, N>& a,
				const smallvec<i32,N>& axes,
				const T initial_value
			)
		{
			ops::_internal::validate_axes(a, axes);
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
				if ( std::find(axes_.begin() , axes_.end() , i) == axes_.end() ) {
					new_shape.append(a.shape()[i]);
					new_str.append(a.strides()[i]);
				}

			Tensor<T, N> result(vec::slice(new_shape, axes.size(), a.rank()));
			if(initial_value)
			  for (auto& item : result)
				item = initial_value;

			Tensor<T, N>projection(new_shape,
					ops::_internal::bcast_strides(result, new_shape)
					, result.offset(), result.data());
			
			return std::make_tuple(Tensor<T, N>(new_shape, new_str,
				a.offset(), a.data()), projection, result);
		}

		

		template<typename T, u32 N, typename Operation>
		Tensor<T, N> _generic_reduction(
			const Tensor<T, N>& A,
			const smallvec<i32,N> axes,
			const Operation& op,
			const bool keepdims = 0 ,
			const T initial_value = 0 //optional param only rly useful for reduce multiply
		)
		{
			auto[permuted,projection,result] = [&A, &axes, &initial_value] {
				if (axes.size() == A.rank())
				{
					Tensor<T, N> result({ 1 });
					result[0] = initial_value;
					return std::make_tuple(A, ops::broadcast_to(result, A.shape()), result);
				}
				return _reduce(A, axes, initial_value);
			} ();
			 
			smallvec<size_t, N> indices(A.rank());
			ops::_internal::_element_wise(permuted, projection, projection, op, indices);

			if (keepdims) {
				smallvec<size_t, N> keepdims = A.shape();
				for (auto i : axes)
					keepdims[i] = 1;
				result = result.reshape(keepdims);
			}

			return result;
		}

	};

	template<typename T, u32 N>
	T sum(const beta::Tensor<T, N>& a)
	{
		T s = 0;
		for (const auto& item : a)
			s += item;
		return s;
	}
	
	
	template<typename T , u32 N>
	Tensor<T, N> sum(const Tensor<T, N>& a, const smallvec<i32, N>& axes,
		const bool keepdims = 0)
	{
		return _internal::_generic_reduction(a, axes, std::plus{}, keepdims);
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
	beta::Tensor<T, N> multiply(const beta::Tensor<T, N>& a, std::initializer_list<i32> axes,
		const bool keepdims = 0)
	{
		return _internal::_generic_reduction(a, axes, std::multiplies{}, keepdims , 1);
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
	beta::Tensor<T, N> mean(const beta::Tensor<T, N>& a, const smallvec<i32>& axes,
		const bool keepdims = 0)
	{
		if (axes.size() == a.rank()) {
			ops::_internal::validate_axes(a, axes);
			Tensor<T, N> ret({ 1 });
			ret[0] = mean(a);
			return ret;
		}
		// Compute the sum of the elements along the specified axis
		auto total = sum(a, axes, keepdims);

		// Compute the size of the reduction dimension
		T prod = 1;
		for (auto i : axes)
			prod *= a.shape()[i];

		// Divide the sum by the size of the reduction dimension to get the mean
		total /= prod;
		return total;
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
	Tensor<T, N> variance(const Tensor<T, N>& a, const std::initializer_list<i32> axes,
		const bool keepdims=0)
	{
		if (axes.size() == a.rank()) {
			ops::_internal::validate_axes(a, axes);
			Tensor<T, N> ret({ 1 });
			ret[0] = variance(a);
			return ret;
		}
		T prod = 1;
		for (auto i : axes) {
			prod *= a.shape()[i];
		};

		Tensor<T,N> mean_ = mean(a , axes , 1);


		Tensor<T, N> x = ops::Subtract(a, mean_ );

		x.apply([](T item) {return item * item; });

		Tensor<T, N> ret = sum(x, { axes }, keepdims);

		ret /= prod;

		return ret;

	}

	

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
		if (axes.size() == a.rank()) {
			ops::_internal::validate_axes(a, axes);
			Tensor<T, N> ret({ 1 });
			ret[0] = stddev(a);
			return ret;
		}
		Tensor<T, N> var = variance(a, axes , keepdims);
		var.apply([](T item) {return std::sqrt(item); });
		return var;
	};

	template<typename T, u32 N >
	T norm(const Tensor<T, N>& a)
	{
		T norm_ = 0;
		for (const auto& item : a)
			norm_ += item * item;
		return std::sqrt(norm_);

	}

	template<typename T, u32 N>
	Tensor<T, N> norm(const Tensor<T, N>& a, const smallvec<i32,N>& axes,
		const bool keepdims =0)
	{
		return reduce::_internal::_generic_reduction(a, axes,
			[](T a, T b) {return b + (a * a); }, keepdims)
			.apply([](f32 x) {return std::sqrt(x); });
	}
 


};