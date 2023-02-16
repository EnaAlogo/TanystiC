#pragma once
#include "init.h"
#include "smallvec.hpp"
#include "TensorOps.hpp"

#include "TensorIterators.hpp"




namespace cum {
#if 0
	template<typename T, u32 N >
	T multiply(const Tensor<T, N>& a)
	{
		T result = 1;
		ContiguousIteration(a,
			{
			[&](T tensor_item) ->void
			{
				result *= tensor_item;
			}
		}
		);
		return result;
	}
	template<typename T, u32 N >
	Tensor<T, N > multiply(const Tensor<T, N>& a, i32 axis)
	{
		axis = negative_index(axis, N);
		if (axis >= N)
			throw std::invalid_argument("axis value larger than num of dimensions");
		Tensor<T, N> result(a.shape());

		iter_with_indices(result,
			
		{
			[&](T& item, const std::array<size_t, N>& indices)
			{
				size_t f = flat_index_array(result._strides(), result.shape(), indices);
				item = a[f];
				if (indices[axis]!=0) {
					item *= result[ f - result._strides()[axis] ];
				}

			}
		});
		return result;
	}

	template<typename T, u32 N>
	Tensor<T, N > sum(const Tensor<T, N>& a, i32 axis)
	{
		axis = negative_index(axis, N);
		if (axis >= N)
			throw std::invalid_argument("axis value larger than num of dimensions");
		Tensor<T, N> result(a.shape());
		iter_with_indices(result,
		
		{
			[&](T& item, const std::array<size_t, N>& indices)
			{
				size_t f = flat_index_array(result._strides(), result.shape(), indices);
				item = a[f];
				if (indices[axis] != 0) {
					item *= result[f - result._strides()[axis]];
				}

			}
		});
		return result;
	}

	template<typename T, u32 N >
	T sum(const Tensor<T, N>& a)
	{
		T result = 1;
		ContiguousIteration(a,
		{
			[&](T tensor_item) ->void
			{
				result += tensor_item;
			}
		}
		);
		return result;
	}
	template<typename T, u32 N>
	Tensor<T, N > mean(const Tensor<T, N>& a, i32 axis)
	{
		// Compute the sum of the elements along the specified axis
		Tensor<T, N > sum = cum::sum(a, axis);
		// Compute the size of the reduction dimension
		size_t dim_size = a.shape()[axis];
		// Divide the sum by the size of the reduction dimension to get the mean
		sum /= dim_size;
		return sum;
	}
	template<typename T, u32 N>
	T mean(const Tensor<T, N>& a)
	{
		// Compute the sum of the elements along the specified axis
		T sum = cum::sum(a);
		// Compute the size of the reduction dimension
		size_t size = a.size();
		// Divide the sum by the size of the reduction dimension to get the mean
		return sum / size;
	}


#endif
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

#if 0

	template<typename T, u32 N, typename =
		std::enable_if_t< (N > 1) >>
		std::tuple<Tensor<T, N - 1>, Tensor<T, N>, std::array<size_t, N>,
		std::array<size_t, N >> _set_up_reduction(const Tensor<T, N>& a, i32 axis,
			T initial_value)
	{
		axis = negative_index(axis, N);
		if (axis > N)
			throw std::invalid_argument("Invalid axis argument out of bounds");

		std::array<size_t, N> new_shape = { 0 };
		std::array<size_t, N > new_strides = { 0 };

		new_shape[0] = a.shape()[axis];
		new_strides[0] = a._strides()[axis];
		u32 k = 1;
		for (int i = 0; i < N; i++)
			if (i != axis) {
				new_shape[k] = a.shape()[i];
				new_strides[k] = a._strides()[i];
				k++;
			}

		std::array<size_t, N - 1> result_shape;
		std::copy(new_shape.begin() + 1, new_shape.end(), result_shape.begin());

		Tensor<T, N - 1> result(result_shape);

		for (int i = 0; i < result.size(); i++)
			result[i] = initial_value;

		Tensor<T, N> it = a;
		it._unsafe_set_shape() = new_shape;
		it._unsafe_set_strides() = new_strides;
		Tensor<T, N > t = broadcast_to(result, it.shape());
		std::array<size_t, N> broadcast_shape = t.shape();
		std::array<size_t, N>broadcast_strides = t._strides();
		return { result , it ,broadcast_shape  ,broadcast_strides };
	}


	template<typename T, u32 N, typename =
		std::enable_if_t< (N > 1) > >
		Tensor<T, N - 1> multiply(const Tensor<T, N>& a, i32 axis = 0,
			T initial_value = 1)
	{
		auto [result, iterate, bcast_shape, bcast_strides] = _set_up_reduction(a, axis, initial_value);
		iter_with_indices(iterate,

			{
				[&](T item, const std::array<size_t, N>& indices)
				{
					size_t flat_index = flat_index_array(bcast_strides, bcast_shape, indices);
					result[flat_index] *= item;
				}
			});
		return result;
	}

	template<typename T >
	T multiply(const Vector<T>& a, T initial_value = 1)
	{

		for (size_t i = 0; i < a.size(); i++)
			initial_value *= a[i * a._strides()[0]];

		return initial_value;
	}

	template<typename T, u32 N, typename =
		std::enable_if_t< (N > 1) >>
		Tensor<T, N - 1> sum(const Tensor<T, N>& a, i32 axis = 0,
			T initial_value = 0)
	{
		auto [result, iterate, bcast_shape, bcast_strides] = _set_up_reduction(a, axis, initial_value);
		iter_with_indices(iterate,

			{
				[&](T item, const std::array<size_t, N>& indices)
				{
					size_t flat_index = flat_index_array(bcast_strides, bcast_shape, indices);
					result[flat_index] += item;
				}
			});
		return result;
	}

	template<typename T >
	T sum(const Vector<T>& a, T initial_value = 1)
	{

		for (size_t i = 0; i < a.size(); i++)
			initial_value += a[i * a._strides()[0]];

		return initial_value;
	}

	template<typename T, u32 N, typename =
		std::enable_if_t< (N > 1) >>
		Tensor<T, N - 1> mean(const Tensor<T, N>& a, i32 axis = 0)
	{
		// Compute the sum of the elements along the specified axis
		Tensor<T, N - 1> sum = sum(a, axis);
		// Compute the size of the reduction dimension
		size_t dim_size = a.shape()[axis];
		// Divide the sum by the size of the reduction dimension to get the mean
		sum /= dim_size;
		return sum;
	}
	template<typename T>
	T mean(const Vector<T>& a)
	{
		// Compute the sum of the elements along the specified axis
		T sum = sum(a);
		// Compute the size of the reduction dimension
		size_t size = a.size();
		// Divide the sum by the size of the reduction dimension to get the mean
		return sum / size;
	}
#endif
	namespace _internal {
		template<typename T, u32 N>
		std::tuple<beta::Tensor<T, N>, beta::Tensor<T, N>, smallvec<size_t, N>>
			set_up_reduction(
				const beta::Tensor<T, N>& a,
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
				if (!axes_.contains(i)) {
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
		template<typename T, u32 N>
		std::tuple<beta::Tensor<T, N>, beta::Tensor<T, N>, smallvec<size_t, N>>
			set_up_reduction(
				const beta::Tensor<T, N>& a,
				std::initializer_list<i32> axes,
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
				if (!axes_.contains(i)) {
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
		if (axes.size() == a.rank()) {
			ops::_internal::validate_axes(a, axes);
			Tensor<T, N> ret({ 1 });
			ret[0] = sum(a);
			return ret;
		}
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
	beta::Tensor<T, N> sum(const beta::Tensor<T, N>& a, std::initializer_list<i32> axes,
		const bool keepdims = 0 )
	{
		if (axes.size() == a.rank()) {
			ops::_internal::validate_axes(a, axes);
			Tensor<T, N> ret({ 1 });
			ret[0] = sum(a);
			return ret;
		}
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
		if (axes.size() == a.rank()) {
			ops::_internal::validate_axes(a, axes);
			Tensor<T, N> ret({ 1 });
			ret[0] = multiply(a);
			return ret;
		}
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
	beta::Tensor<T, N> mean(const beta::Tensor<T, N>& a, std::initializer_list<i32> axes,
		const bool keepdims =0 )
	{
		if (axes.size() == a.rank()) {
			ops::_internal::validate_axes(a, axes);
			Tensor<T, N> ret({ 1 });
			ret[0] = mean(a);
			return ret;
		}
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
	Tensor<T, N> variance(const Tensor<T, N>& a, std::initializer_list<i32> axes,
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

		Tensor<T, N> ret = sum(x, axes , keepdims);

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
	Tensor<T, N> norm(const Tensor<T, N>& a, std::initializer_list<i32> axes,
		const bool keepdims =0)
	{
		if (axes.size() == a.rank()) {
			ops::_internal::validate_axes(a, axes);
			Tensor<T, N> ret({ 1 });
			ret[0] = norm(a);
			return ret;
		}
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

	

};