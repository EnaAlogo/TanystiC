#pragma once
#include "init.h"
#include "TensorIterators.hpp"
#include "initializers.hpp"
#include "SharedPtr.hpp"
#include "ScalarOps.hpp"
#include "smallvec.hpp"


namespace ops
{
	template<typename T , u32 N = 7 > 
	using Tensor = beta::Tensor<T, N>;

	template<typename T>
	using Matrix = Tensor<T, 2>;

	template<typename T>
	using Vector = Tensor<T, 1>;

	template<typename T, u32 N, u32 M>
	beta::Tensor<T, M > broadcast_to(const beta::Tensor<T, N>& a, const smallvec<size_t, M>& shape)
	{
		if (!(shape.size() >= a.rank()))
			std::cerr << "shape " << a.shape() << " cannot be broadcasted to " << shape
			, throw std::invalid_argument("");
		smallvec<size_t, M > bcastable_shape;
		for (i32 i = 0; i < shape.size() - a.rank(); i++)
			bcastable_shape.append(1);
		for (const size_t i : a.shape())
			bcastable_shape.append(i);

		smallvec<size_t, M> rstrides = vec::compute_strides(bcastable_shape,
			a.strides()[-1]);

		for (int i = 0; i < shape.size(); i++) {
			if (bcastable_shape[i] != shape[i] && bcastable_shape[i] == 1)
				rstrides[i] = 0;
			else if (bcastable_shape[i] != shape[i])
				std::cerr << "shape " << a.shape() << " cannot be broadcasted to " << shape
				, throw std::invalid_argument("");
			bcastable_shape[i] = shape[i];
		}
		return beta::Tensor<T, M>(bcastable_shape, rstrides, a.offset(), a.data());
	}


	namespace _internal
	{
		template<u32 N , u32 M>
		bool is_broadcastable(const smallvec<size_t, N>& a, const smallvec<size_t, M>& b)
		{
			uint small = std::min(a.size(), b.size());
			for (i32 i = 1 ; i < small + 1 ; ++i)
			{
				if (a[-i] != b[-i] && (a[-i] != 1 && b[-i] != 1))
					return false;
			}
			return true;
		}
		template<u32 N, u32 N1>
		smallvec<size_t, N* (N > N1) + N1 * (N <= N1)> find_broadcast_shape(const smallvec<size_t, N >& right, const smallvec<size_t, N1>& left)
		{
			smallvec<size_t, N* (N > N1) + N1 * (N <= N1)> result(std::max(right.size() , left.size()));
			auto first_part = [&](const auto& big, const auto& small) ->void
			{
				for (i32 i = 1; i <= small.size() ; ++i)
				{
					i32 b_i = big.size() - i, s_i = small.size() - i;
					assert(small[s_i] == big[b_i] || big[b_i] == 1 || small[s_i] == 1 && "Non broadcastable shapes");
					result[b_i] = std::max(big[b_i], small[s_i]);
				}
			};
			auto second_part = [&](const auto& leftover , size_t size_) ->void
			{
				for (i32 i = 0; i < size_; ++i)
					result[i] = leftover[i];
			};
			if (right.size() > left.size())
			{
				first_part(right, left);
				second_part(right, right.size() - left.size());
			}
			else 
			{
				first_part(left, right);
				second_part(left, left.size() - right.size());
			}
			return result;
		}
		template<typename T,u32 M, u32 N>
		smallvec<size_t , N*(N>M) + M*(N<=M)> bcast_strides(
			const beta::Tensor<T,M>& a, const smallvec<size_t, N>& shape)
		{			
			smallvec<size_t, N* (N > M) + M * (N <= M)> tmp;
			for (i32 i = 0; i < (i32)shape.size() - (i32)a.rank(); i++)
				tmp.append(1);
			for (const size_t item : a.shape())
				tmp.append(item);
			smallvec<size_t, N* (N > M) + M * (N <= M)> 
				strides = vec::compute_strides(tmp, a.strides().back());
			for (i32 i = 0; i < shape.size(); i++)
				if (tmp[i] != shape[i])
					strides[i] = 0;
			return strides;
		}

	
		template<typename T, u32 N>
		void validate_axes(const Tensor<T, N>& a, const smallvec<i32,N>&axes)
		{
			if (axes.size() > a.rank())
				std::cerr << "axes cannot be more than rank of tensor", throw std::invalid_argument("");
			auto contains = [&a](const auto& arr)
			{
				for (i32 el : arr)
					if (el > 1)
						std::cerr << "duplicate axes not allowed", throw std::invalid_argument("");
			};
			std::array<i32, N> uniq = { 0 };
			for (i32 i = 0; i < axes.size(); i++)
			{
				i32 ax = axes[i];
				ax = ax >= 0 ? ax : ax + a.rank();
				if (ax > a.rank())
					std::cerr << "axis out of bounds", throw std::invalid_argument("");
				uniq[ax] ++;
			}
			contains(uniq);
		}
		template<typename T , u32 N>
		void validate_axes(const Tensor<T, N>& a, const std::initializer_list<i32>axes)
		{
			if (axes.size() > a.rank())
				std::cerr << "axes cannot be more than rank of tensor", throw std::invalid_argument("");
			auto contains = [&a](const auto& arr)
			{
				for (i32 el : arr)
					if (el > 1)
						std::cerr << "duplicate axes not allowed", throw std::invalid_argument("");
			};
			std::array<i32, N> uniq = { 0 };
			for (i32 i = 0; i < axes.size(); i++)
			{
				i32 ax = *(axes.begin() + i);
				ax = ax >= 0 ? ax : ax + a.rank();
				if (ax > a.rank())
					std::cerr << "axis out of bounds", throw std::invalid_argument("");
				uniq[ax] ++;
			}
			contains(uniq);
		}

		template<typename T , u32 N>
		auto validate_and_getReversed_view(const beta::Tensor<T, N>& tensor, const std::initializer_list<i32> axes)
		{
			auto reverse = [](
				const beta::Tensor<T, N>& conv,
				const std::initializer_list<i32> axes)
			{
				class : public beta::Tensor<T, N>
				{
				public:

					using beta::Tensor<T, N>::Tensor;

					constexpr size_t calc_offset(const smallvec<i64, N>& indices) const override
					{
						smallvec<i64, N> bindices = indices;
						for (size_t axis : axes)
							bindices[axis] = this->shape()[axis] - 1 - bindices[axis];
						return vec::compute_flat_index(
							this->strides(), this->shape(), bindices
						);
					};
					constexpr size_t calc_offset(const smallvec<size_t, N>& indices) const override
					{
						smallvec<size_t, N> bindices = indices;
						for (size_t axis : axes)
							bindices[axis] = this->shape()[axis] - 1 - bindices[axis];
						return vec::compute_flat_index(
							this->strides(), this->shape(), bindices
						);
					};

					constexpr  smallvec<size_t, N>& set_axes()
					{
						return axes;
					}
				private:
					smallvec<size_t, N> axes;

				}ret{ conv.shape(),conv.strides(),conv.offset(),conv.data() };

				if (axes.size() == 0)
				{
					ret.set_axes() = vec::tovec<size_t, N>(range(ret.rank()));
				}
				else
				{
					smallvec<size_t, N> axes_toreverse;
					for (i32 axis : axes)
						axes_toreverse.append(axis >= 0 ? axis : axis + ret.rank());
					ret.set_axes() = axes_toreverse;
				}
				return ret;
			};

			validate_axes(tensor , axes);

			return reverse(tensor, axes);
		};

		template<u32 N>
		decltype(auto) _element_wise_offsets
		(const smallvec<size_t, N>& a, const smallvec < size_t, N>& b,
			const smallvec < size_t, N>& c, const smallvec < size_t, N>& i)
		{
			//assert(a.size() == b.size() && c.size() == b.size() && a.size() == i.size());
			size_t ai = 0, bi = 0, ci = 0;
			for (size_t k = 0; k < a.size(); ++k)
			{
				ai += a[k] * i[k];
				bi += b[k] * i[k];
				ci += c[k] * i[k];
			}
			struct { size_t x, y, z; }ret{ ai,bi,ci };
			return ret;
		};

		template<typename T, u32 N, typename Operation>
		void _element_wise(const Tensor<T, N>& a, const Tensor<T, N>& b, Tensor<T, N>& c,
			const Operation& op, smallvec<size_t, N>& indices, i32 level = 0)
		{
			if (level == a.rank() - 1)
			{
				for (size_t i = 0; i < a.shape().back(); ++i)
				{
					indices[level] = i;
					auto [ai, bi, ci] = _element_wise_offsets(a.strides(), b.strides(), c.strides(), indices);
					c[ci] = op(a[ai], b[bi]);
				}
				indices[level] = 0;
				return;
			}
			for (size_t i = 0; i < a.shape()[level]; ++i) {
				indices[level] = i;
				_element_wise(a, b, c, op, indices, level + 1);
			}
			indices[level] = 0;
		}
		
		template<typename T, u32 N, u32 M, u32 K, typename Operation>
		void binary_operation(
			const Tensor<T, N>& a,
			const Tensor < T, M>& b,
			Tensor<T, K>& c,
			const Operation& op)
		{

			Tensor<T, K> aa = ops::broadcast_to(a, c.shape());
			Tensor<T, K> bb = ops::broadcast_to(b, c.shape());
			smallvec<size_t, K> indices(c.rank());
			_element_wise(aa, bb, c, op, indices);
		}

		template<typename T, u32 N, u32 M, typename Operation>
		decltype(auto) binary_operation(
			const Tensor<T, N>& a,
			const Tensor < T, M>& b,
			const Operation& op)
		{
			using tensor_t = std::conditional< (N > M), Tensor<T, N>, Tensor<T, M> >::type;

			tensor_t c(ops::_internal::find_broadcast_shape(a.shape(), b.shape()));
			binary_operation(a, b, c, op);
			return c;
		}

	};//end internal

	template<typename T, u32 N ,u32 M >
	beta::Tensor<T> Add(const beta::Tensor<T, N>& a, const beta::Tensor<T, M>& b)
	{
		return _internal::binary_operation(a, b, _add<T>{});
	}
	template<typename T, u32 N, u32 M >
	beta::Tensor<T> Subtract(const beta::Tensor<T, N>& a, const beta::Tensor<T, M>& b)
	{
		return _internal::binary_operation(a, b, _sub<T>{});
	}
	template<typename T, u32 N, u32 M >
	beta::Tensor<T> Multiply(const beta::Tensor<T, N>& a, const beta::Tensor<T, M>& b)
	{
		return _internal::binary_operation(a, b, _mul<T>{});
	}
	template<typename T, u32 N, u32 M >
	beta::Tensor<T> Divde(const beta::Tensor<T, N>& a, const beta::Tensor<T, M>& b)
	{
		return _internal::binary_operation(a, b, _div<T>{});
	}
	template<typename T, u32 N, u32 M ,u32 K>
	void Add(
		const beta::Tensor<T, N>& a,
		const beta::Tensor<T, M>& b,
		beta::Tensor<T,K>& out)
	{
		   _internal::binary_operation(a, b, out, _add<T>{} );
	}
	template<typename T, u32 N, u32 M ,u32 K>
	void Subtract(
		const beta::Tensor<T, N>& a,
		const beta::Tensor<T, M>& b,
		beta::Tensor<T, K>& out)
	{
			_internal::binary_operation(a, b, out,_sub<T>{});
	}
	template<typename T, u32 N, u32 M , u32 K >
	void Multiply(
		const beta::Tensor<T, N>& a,
		const beta::Tensor<T, M>& b,
		beta::Tensor<T, K>& out)
	{
			_internal::binary_operation(a, b, out, _mul<T>{});
			
	}
	template<typename T, u32 N, u32 M ,u32 K>
	void Divde(
		const beta::Tensor<T, N>& a,
		const beta::Tensor<T, M>& b,
		beta::Tensor<T, K>& out)
    {
			_internal::binary_operation(a, b,  out , _div<T>{} );
	}

	template<typename value_type, u32 N  >
	beta::Tensor<value_type, N> transpose(const beta::Tensor<value_type, N>& tensor,
		std::initializer_list<int32_t> perms = {}) {
		if (tensor.rank() == 1)
			return tensor;
		smallvec<i32, N> permutations;

		if (perms.size() != 0) {
			auto validate_perms = [&]()
			{
				smallvec<i32, N> count(perms.size());
				for (i32 perm : perms)
					count[perm]++;
				return count.max() == 1;
			};
			assert(perms.size() == tensor.rank() && "Invalid axes argument");
			if (!validate_perms())
				throw std::invalid_argument("Invalid axes argument");

			permutations.resize(perms.size());
			std::move(perms.begin(), perms.end(), permutations.begin());
		}
		else {
			permutations.resize(tensor.rank());
			for (i32 i = tensor.rank() - 1; i > -1; --i)
				permutations[tensor.rank() - 1 - i] = i;
		}
		smallvec<size_t, N> shape(tensor.rank());
		smallvec<size_t, N> strides(tensor.rank());
		for (i32 i = 0; i < (i32)tensor.rank(); ++i)
		{
			shape[i] = tensor.shape()[permutations[i]];
			strides[i] = tensor.strides()[permutations[i]];
		}

		return beta::Tensor<value_type, N>(shape, strides, tensor.offset(), tensor.data());
	}
	template<typename T, u32 N>
	beta::Tensor<T, N> squeeze(const beta::Tensor<T, N>& a)
	{
		smallvec<size_t, N> new_shape;
		for (const size_t el : a.shape())
			if (el != 1)
				new_shape.append(el);
		return a.reshape(new_shape);
	}

	template<typename value_type, u32 N>
	beta::Tensor<value_type, N> transpose(const beta::Tensor<value_type, N>& tensor,
		const smallvec<size_t>& perms)
	{
		if (tensor.rank() == 1)
			return tensor;
		auto validate_perms = [&tensor , &perms]()
		{
			smallvec<i32, N> count(tensor.rank());
			for (i32 perm : perms)
				count[perm]++;
			return count.max() == 1;
		};
		assert(validate_perms());
		smallvec<size_t, N> shape = tensor.shape();
		smallvec<size_t, N> strides = tensor.strides();
		for (auto [i, perm] : enumerate(perms))
		{
			shape[i] = tensor.shape()[perm];
			strides[i] = tensor.strides()[perm];
		}

		return beta::Tensor<value_type, N>(shape, strides, tensor.offset(), tensor.data());
	}
	

	template<typename T, u32 N>
	Matrix<T> matrix_diag(const beta::Tensor<T, N>& a, i32 k = 0)
	{
		/// <summary>
		/// creates a matrix with the kth diagonal containing the elements
		/// of a given tensor (must be 1D) 
		/// </summary>
		/// <param name="a">1D tensor</param>
		/// <param name="k">the diagonal , 0 = main , negative = lowers ,
		/// positive = upper diagonals</param>
		/// <returns>a new matrix with the kth diagonal containing the elements
		/// and everything else 0</returns>
		assert(a.rank() == 1);
		size_t n = a.shape()[0] + std::abs(k);
		auto diag = tensor::zeros<T>(smallvec<size_t ,2>{n, n});
		smallvec<size_t> indices(2);

		if (k >= 0) { indices[0] = 0, indices[1] = k; }
		else { indices[0] = -k, indices[1] = 0; }

		auto ifunc = [&](i64 i) {return (diag.strides()[0] * (i + indices[0]))
			+ (diag.strides()[1] * (i + indices[1])); };

		for (i64 i = 0; i < a.shape()[0]; i++)
		{
			size_t j = ifunc(i);
			diag[j] = a[i];
		}
		return diag;
	}
	template<typename T, u32 N>
	Vector<T> diagonal(const beta::Tensor<T, N>& a,i32 offset =0, i32 axis = 0 , i32 axis1=1)
	{
		/// <summary>
		/// produces a vector with the kth diagonal of a given matrix
		/// if the input is a tensor of ndim >2 you can pick which axes 
		/// you want the diagonal from defaults to 0,1 for a the most usual
		/// 2D matrix case
		/// </summary>
		/// <param name="a">a tensor</param>
		/// <param name="offset">the diagonal eg 0 = main , negative = lower ,
		///  positive = upper diagonals</param>
		/// <param name="axis">optional parameter in the case that the input
		/// is a tensor of more than 2 D</param>
		/// <param name="axis1">optional parameter in the case that the input
		/// is a tensor of more than 2 D</param>
		/// <returns>a 1D vector that contains the elements of the chosen diagonal
		/// of the given tensor</returns>
		assert(a.rank() >= 2);
		axis = axis >= 0 ? axis : axis + a.rank();
		axis1 = axis1 >= 0 ? axis1 : axis1 + a.rank();
		smallvec<size_t, N> indices(a.rank());
		i64 vsize;//= std::min(a.shape()[axis1], a.shape()[axis]) ;
		if (offset >= 0)
		{
			vsize = std::min((i64)a.shape()[axis], (i64)a.shape()[axis1] - offset);
			for (const auto [i, dim] : enumerate(a.shape()))
				indices[i] = i == axis1 ? offset : 0;
		}
		else
		{
			vsize = std::min((i64)a.shape()[axis] + offset, (i64)a.shape()[axis1]);
			for (const auto [i, dim] : enumerate(a.shape()))
				indices[i] = i == axis ? -offset : 0;
		}
		auto ifunc = [&](i64 i)
		{
			i64 prod = 0;
			for (auto [str, ii] : zip(a.strides(), indices))
				prod += str * (ii+i);
			return prod;
		};
		if (vsize <= 0)
			return Vector<T>();
		Vector<T> diag = tensor::ones<T>(smallvec<size_t, 1>{(size_t)vsize});
		for (i64 i = 0; i < vsize; i++)
		{
			size_t ii = ifunc(i);
			diag[i] = a[ii];
		}
		return diag;
	}

	template<typename T , u32 N>
	beta::Tensor<T> diag(const beta::Tensor<T, N>& a, i32 k = 0)
	{
		if (a.rank() == 1)
		{
			auto res = matrix_diag(a, k);
			if (res.rank() == 0)
				return beta::Tensor<T>();
			return res.reshape(smallvec<size_t>{res.shape()[0], res.shape()[1]});
		}
		else if (a.rank() == 2)
		{
			auto res = diagonal(a, k );
			if (res.rank() == 0)
				return beta::Tensor<T>();
			return res.reshape(smallvec<size_t>{res.shape()[0]});
		}
		else
			throw std::invalid_argument("Input must be 1D or 2D");
	}
	
	template<typename T , u32 N>
	beta::Tensor<T ,N> pad(
		const beta::Tensor<T, N>& tensor, 
		const smallvec<std::pair<size_t , size_t>,N>& padding,
		T value = 0)
	{
		assert(padding.size() <= tensor.rank());

		auto compute_output_shape = 
			[](const auto& tensor,const auto& paddings)
			-> std::pair< smallvec<size_t,N> , std::array<i64,N>>
		{
			std::array< i64, N> pads = { 0 };
			smallvec<size_t,N> out_shape(tensor.rank());
			for (i32 i = 0; i < paddings.size(); i++) {
				auto [before, after] = paddings[i];
				out_shape[i] = tensor.shape()[i] + before + after;
				pads[i] = before;
			}
			for (i32 i = paddings.size(); i < tensor.rank(); ++i) 
				out_shape[i] = tensor.shape()[i];

	
			return { out_shape , pads };
		};
		
		auto[outshape , shifts ]= compute_output_shape(tensor, padding);
		beta::Tensor<T, N> out_tensor(outshape);

		auto in_tensor_range = 
			[&tensorshape = tensor.shape() , 
			&outshape = std::as_const(outshape),
			&shifts = std::as_const(shifts) ]
		(auto& indices) -> bool
		{
			for (const auto& [i , dim] : enumerate( tensorshape ))
			{
				i64 pad = shifts[i];
				size_t index = indices[i];
				if (index < pad || index >= dim + pad)
					return false;
				else
					indices[i] -= pad;
			}
			return true;
		};
		
		for (auto [i, item] : tEnumerate(out_tensor))
		{
			T pad_or_item = value;
			smallvec<size_t,N>b_indices = i;
			if (in_tensor_range(b_indices))
			{
				
				size_t position = tensor.calc_offset(b_indices);
				pad_or_item = tensor[position];
			}
			item = pad_or_item;
		}
		return out_tensor;
	}

	template<typename T, u32 N>
	beta::Tensor<T, N> pad(
		const beta::Tensor<T, N>& tensor,
		const std::pair<size_t, size_t>& padding,
		T value = 0)
	{
		smallvec < std::pair<size_t, size_t> ,N>paddings(tensor.rank());
		for (i32 i = 0; i < tensor.rank(); i++)
			paddings[i] = padding;
		return pad(tensor, paddings, value);
	}
	template<typename T, u32 N>
	beta::Tensor<T, N> pad(
		const beta::Tensor<T, N>& tensor,
		size_t padding,
		T value = 0)
	{
		smallvec < std::pair<size_t, size_t>, N>paddings(tensor.rank());
		for (i32 i = 0; i < tensor.rank(); i++)
			paddings[i] = { padding,padding };
		return pad(tensor, paddings, value);
	}


	template<typename T, u32 N>
	beta::Tensor<T, N> flip(const beta::Tensor<T, N>& tensor,
		std::initializer_list<i32> axes={})
	{
		/// <summary>
		/// flips the tensor along selected axes, 
		/// as of right now this returns a new tensor 
		/// however the method used to produce it first creates
		/// a flipped view in the future i am looking to make this 
		/// function return a view rather than copying the view to a tensor
		/// but atm the view is of type anonymous class
		/// </summary>
		/// <typeparam name="T">tensor dtype</typeparam>
		/// <typeparam name="N">smallvec max stack size</typeparam>
		/// <param name="tensor">a tensor</param>
		/// <param name="axes">axes to perform the flip on</param>
		/// <returns>a new tensor with the elements along the chose axes flipped</returns>
		auto reversed_view = _internal::validate_and_getReversed_view(tensor, axes);

		beta::Tensor<T, N> out(tensor.shape());

		for (auto [i, rev] : enumerate(reversed_view))
			out[i] = rev;
		return out;
	}


	template<typename T, u32 N>
	void flip(const beta::Tensor<T, N>& tensor,
		  beta::Tensor<T,N>& out , std::initializer_list<i32> axes = {})
	{
		

		if (out.shape() != tensor.shape())
			std::cerr << "invalid out tensor param differs in shapes", throw std::invalid_argument("");

		auto reversed_view = _internal::validate_and_getReversed_view(tensor, axes);

		for (auto [i, rev] : enumerate(reversed_view))
			out[i] = rev;

	}
	
	
	template<typename T,u32 N>
	beta::Tensor<T, N> slice(const beta::Tensor<T,N>& a,std::initializer_list<range> slices)
	{
	/// <summary>
	/// axes should be passed in order if you dont wish to slice an axis
	/// pass an empty range or a {} 
	/// in general when slicing an array following rules apply
	/// BOTH ->
    ///  compute the new shape by doing end - start on the ranges
    ///  unsafely set the new shape
    /// 
    /// IF SLICING TAIL ->
    ///   do not modify the strides just set the new shape
    /// 
    /// IF SLICING HEAD ->
    ///   modify the offset to be ->
    ///     for every n dimensions sliced 
    ///     add the corresponding stride multiplied by
    ///     the number x (x first elements to slice)
	///   do not modify the strides
	/// </summary>
	/// <typeparam name="T">type of the tensor</typeparam>
	/// <typeparam name="N">irrelevant smallvec(dim vector) max stack size</typeparam>
	/// <param name="a"> the tensor </param>
	/// <param name="slices"> list of ranges/ slices</param>
	/// <returns> the sliced tensor without making a copy </returns>
	
		if (slices.size() > a.rank())
			std::cerr << "indices cannot exceed rank in count", throw std::invalid_argument("");

		smallvec<size_t , N > x_shape = a.shape();
		i64 offset = a.offset();

		for (i32 i =0 ; i < slices.size() ; ++i)
		{
			const range& slice = *(slices.begin() + i);
			if (!slice.get_step())
				continue;

			i64 before = slice.get_start(), after = slice.get_end();
			before = before >= 0 ? before : before + x_shape[i];
			after = after >= 0 ? after : after + x_shape[i];

			i64 diff = std::clamp(after - before, 0ll, (i64)x_shape[i]);

			x_shape[i] = static_cast<size_t>(diff);
			offset += static_cast<i64>(a.strides()[i]) * before ;

		}

		return beta::Tensor<T,N>(x_shape, a.strides(), offset, a.data());
	}


};//end ops



namespace math
{
	template<typename T,  u32 N = 7>
	using Tensor = beta::Tensor<T, N>;

	template<typename T>
	using Matrix = Tensor<T, 2>;

	template<typename T>
	using Vector = Tensor<T, 1>;

	namespace _internal {


		template<typename T, u32 N >
		std::tuple<Matrix<T>, smallvec<size_t>, smallvec<size_t>> _tensordot_reshape(
			const beta::Tensor<T, N>& a, const smallvec<size_t>& axes, bool flipped = 0)
		{
			smallvec<size_t> free;
			for (i32 i = 0; i < a.rank(); i++)
				if (!axes.contains(i))
					free.append(i);
			smallvec<size_t> free_dims;
			for (size_t i : free)
				free_dims.append(a.shape()[i]);
			size_t prod_free = free.prod([&](size_t dim)
				{
					return a.shape()[dim];
				});
			size_t prod_axes = axes.prod(
				[&](size_t dim)
				{
					return a.shape()[dim];
				}
			);
			smallvec<size_t> perm = free;
			smallvec<size_t, 2> new_shape;
			if (flipped) {
				perm.prepend(axes);
				new_shape = { prod_axes , prod_free };
			}
			else {
				perm.append(axes);
				new_shape = { prod_free , prod_axes };
			}
			smallvec<size_t> dimrange;
			for (u32 i = 0; i < a.rank(); i++)
				dimrange.append(i);
			beta::Tensor<T, N> a_trans;
			if (perm != dimrange)
				a_trans = ops::transpose(a, perm);
			else
				a_trans = a;
			Matrix<T> reshape_a = a_trans.reshape(new_shape);
			return { reshape_a, free_dims, free_dims };

		}
		template<typename T, u32 N>
		std::tuple<Matrix<T>, smallvec<size_t>, smallvec<size_t>>
			_tensordot_reshape(const beta::Tensor<T, N>& a, size_t axes, bool flipped = 0)
		{
			smallvec<size_t> free;
			for (i32 i = 0; i < a.rank(); i++)
				if (i != axes)
					free.append(i);
			smallvec<size_t> free_dims;
			for (size_t i : free)
				free_dims.append(a.shape()[i]);
			size_t prod_free = free.prod([&](size_t dim)
				{
					return a.shape()[dim];
				});
			size_t prod_axes = a.shape()[axes];
			smallvec<size_t> perm = free;
			smallvec<size_t, 2> new_shape;
			if (flipped) {
				perm.prepend(axes);
				new_shape = { prod_axes , prod_free };
			}
			else {
				perm.append(axes);
				new_shape = { prod_free , prod_axes };
			}
			smallvec<size_t> dimrange;
			for (u32 i = 0; i < a.rank(); i++)
				dimrange.append(i);
			beta::Tensor<T, N> a_trans;
			if (perm != dimrange)
				a_trans = ops::transpose(a, perm);
			else
				a_trans = a;
			Matrix<T> reshape_a = a_trans.reshape(new_shape);

			return std::tuple{ reshape_a, free_dims, free_dims };
		}

		template<typename T, u32 N>
		std::pair<size_t, size_t> _tensordot_axes(const beta::Tensor<T, N>& a,
			const std::array<i32, 2>& axes)
		{
			i32 a_axes = axes[0] >= 0 ? axes[0] : axes[0] + a.rank();
			i32 b_axes = axes[1] >= 0 ? axes[1] : axes[1] + a.rank();
			return { a_axes , b_axes };
		}
		template<typename T, u32 N>
		std::pair<smallvec<size_t>, smallvec<size_t>> 
			_tensordot_axes(const beta::Tensor<T, N>& a, i32 axes)
		{

			assert(axes < a.rank() && axes >= 0 && "invalid axis arg");
			return { vec::tovec<size_t>(range(a.rank() - axes , a.rank()))
					, vec::tovec<size_t>(range(axes)) };

		}


		template<typename T, u32 N>
		std::pair<smallvec<size_t>, smallvec<size_t>> _tensordot_axes(const beta::Tensor<T, N>& a,
			const std::pair<smallvec<i32>, smallvec<i32>>& axes)
		{
			assert(axes.first.size() == axes.second.size());
			smallvec<size_t> a_axes;
			smallvec<size_t> b_axes;
			for (auto& i : axes.first)
				a_axes.append(i >= 0 ? i : i + a.rank());
			for (auto& i : axes.second)
				b_axes.append(i >= 0 ? i : i + a.rank());

			return { a_axes , b_axes };
		}

	}; //end interal

	template<typename T>
	T inner(const Vector<T>& a, const Vector<T>& b)
	{
		assert(a.size() == b.size());
		T prod = 0;
		for (const auto& [l, r] : zip(a, b))
			prod += l * r;
		return prod;
	}

	template<typename T>
	Matrix<T> mat_mul(const Matrix<T>& A, const Matrix<T>& B)
	{
		if (A.shape()[1] != B.shape()[0])
			throw std::invalid_argument("Shape mismatch");

		Matrix<T> prod = tensor::zeros<T>(smallvec<size_t,2>{A.shape()[0] , B.shape()[1]});
		for (size_t i = 0; i < A.shape()[0]; ++i)
			for (size_t j = 0; j < B.shape()[1]; ++j)
				for (size_t l = 0; l < B.shape()[0]; ++l)
					prod(i, j) += A(i, l) * B(l, j);
		return prod;
	}
	template<typename T>
	Matrix<T>& mat_mul(const Matrix<T>& A, const Matrix<T>& B , Matrix<T>&out)
	{
		if (A.shape()[1] != B.shape()[0] || 
			out.shape()[0]!=A.shape()[0] || 
			out.shape()[1]!=B.shape()[1])
			throw std::invalid_argument("Shape mismatch");
		for (size_t i = 0; i < A.shape()[0]; ++i)
			for (size_t j = 0; j < B.shape()[1]; ++j)
				for (size_t l = 0; l < B.shape()[0]; ++l)
					out(i, j) += A(i, l) * B(l, j);
		return out;
	}
	template<typename T , u32 N , u32 M >
	void mat_mul(const beta::Tensor<T,N>& A, const beta::Tensor<T,M>& B, beta::Tensor<T>& outmat)
	{
		smallvec<size_t, 2> output_shape{ A.shape()[0] , B.shape()[1] };
		if (A.shape()[1] != B.shape()[0] || outmat.shape() != output_shape)
			throw std::invalid_argument("Shape mismatch");

		for (size_t i = 0; i < A.shape()[0]; ++i)
			for (size_t j = 0; j < B.shape()[1]; ++j)
				for (size_t l = 0; l < B.shape()[0]; ++l)
					outmat(i, j) += A(i, l) * B(l, j);
	}

	
	template<typename T, u32 N, u32 M>
	beta::Tensor<T> tensordot(const beta::Tensor<T, N>& a, const beta::Tensor<T, M>& b,
		const std::array<i32, 2>& axes)
	{
		auto [a_axes, b_axes] = _internal::_tensordot_axes(a, axes);
		auto [a_reshape, a_free_dims, a_static] = _internal::_tensordot_reshape(a, a_axes);
		auto [b_reshape, b_free_dims, b_static] = _internal::_tensordot_reshape(b, b_axes, true);
		decltype(auto) ction = vec::concat(a_free_dims, b_free_dims);
		Matrix<T> ab_matmul = mat_mul(a_reshape, b_reshape);
		return ab_matmul.reshape(ction);
	}
	template<typename T, u32 N, u32 M>
	beta::Tensor<T> tensordot(const beta::Tensor<T, N>& a, const beta::Tensor<T, M>& b,
		i32 axes)
	{
		auto [a_axes, b_axes] = _internal::_tensordot_axes(a, axes);
		auto [a_reshape, a_free_dims, a_static] = _internal::_tensordot_reshape(a, a_axes);
		auto [b_reshape, b_free_dims, b_static] = _internal::_tensordot_reshape(b, b_axes, true);
		decltype(auto) ction = vec::concat(a_free_dims, b_free_dims);
		Matrix<T> ab_matmul = mat_mul(a_reshape, b_reshape);
		return ab_matmul.reshape(ction);
	}
	template<typename T, u32 N, u32 M>
	beta::Tensor<T> tensordot(const beta::Tensor<T, N>& a, const beta::Tensor<T, M>& b,
		const std::pair<smallvec<i32>, smallvec<i32> >& axes)
	{
		auto [a_axes, b_axes] = _internal::_tensordot_axes(a, axes);
		auto [a_reshape, a_free_dims, a_static] = _internal::_tensordot_reshape(a, a_axes);
		auto [b_reshape, b_free_dims, b_static] = _internal::_tensordot_reshape(b, b_axes, true);
		decltype(auto) ction = vec::concat(a_free_dims, b_free_dims);
		Matrix<T> ab_matmul = mat_mul(a_reshape, b_reshape);
		return ab_matmul.reshape(ction);
	}

	template<typename T , u32 N>
	beta::Tensor<T> exp(const beta::Tensor<T, N>& x)
	{
		beta::Tensor<T, N> ret(x.shape());
		for (const auto [i, item] : enumerate(x))
			ret[i] = std::exp(item);
		return ret;
	}
	template<typename T, u32 N>
	beta::Tensor<T> log(const beta::Tensor<T, N>& x)
	{
		beta::Tensor<T, N> ret(x.shape());
		for (const auto [i, item] : enumerate(x))
			ret[i] = std::log( item );
		return ret;
	}
	template<typename T, u32 N>
	beta::Tensor<T> log2(const beta::Tensor<T, N>& x)
	{
		beta::Tensor<T, N> ret(x.shape());
		for (const auto [i, item] : enumerate(x))
			ret[i] = std::log2(item);
		return ret;
	}
	template<typename T, u32 N>
	beta::Tensor<T> sqrt(const beta::Tensor<T, N>& x)
	{
		beta::Tensor<T, N> ret(x.shape());
		for (const auto [i, item] : enumerate(x))
			ret[i] = std::sqrt(item);
		return ret;
	}
	template<typename T, u32 N>
	beta::Tensor<T> cos(const beta::Tensor<T, N>& x)
	{
		beta::Tensor<T, N> ret(x.shape());
		for (const auto [i, item] : enumerate(x))
			ret[i] = std::cos(item);
		return ret;
	}
	template<typename T, u32 N>
	beta::Tensor<T> sin(const beta::Tensor<T, N>& x)
	{
		beta::Tensor<T, N> ret(x.shape());
		for (const auto [i, item] : enumerate(x))
			ret[i] = std::sin(item);
		return ret;
	}
	template<typename T, u32 N>
	beta::Tensor<T> tanh(const beta::Tensor<T, N>& x)
	{
		beta::Tensor<T, N> ret(x.shape());
		for (const auto [i, item] : enumerate(x))
			ret[i] = std::tanh(item);
		return ret;
	}
	template<typename T, u32 N>
	beta::Tensor<T> atanh(const beta::Tensor<T, N>& x)
	{
		beta::Tensor<T, N> ret(x.shape());
		for (const auto [i, item] : enumerate(x))
			ret[i] = std::atanh(item);
		return ret;
	}
	template<typename T, u32 N>
	beta::Tensor<T> acos(const beta::Tensor<T, N>& x)
	{
		beta::Tensor<T, N> ret(x.shape());
		for (const auto [i, item] : enumerate(x))
			ret[i] = std::acos(item);
		return ret;
	}
	template<typename T, u32 N>
	beta::Tensor<T> asin(const beta::Tensor<T, N>& x)
	{
		beta::Tensor<T, N> ret(x.shape());
		for (const auto [i, item] : enumerate(x))
			ret[i] = std::asin(item);
		return ret;
	}
	template<typename T, u32 N>
	beta::Tensor<T> tan(const beta::Tensor<T, N>& x)
	{
		beta::Tensor<T, N> ret(x.shape());
		for (const auto [i, item] : enumerate(x))
			ret[i] = std::tan(item);
		return ret;
	}
	template<typename T, u32 N>
	beta::Tensor<T> atan(const beta::Tensor<T, N>& x)
	{
		beta::Tensor<T, N> ret(x.shape());
		for (const auto [i, item] : enumerate(x))
			ret[i] = std::atan(item);
		return ret;
	}
	template<typename T, u32 N>
	beta::Tensor<T> sinh(const beta::Tensor<T, N>& x)
	{
		beta::Tensor<T, N> ret(x.shape());
		for (const auto [i, item] : enumerate(x))
			ret[i] = std::sinh(item);
		return ret;
	}
	template<typename T, u32 N>
	beta::Tensor<T> asinh(const beta::Tensor<T, N>& x)
	{
		beta::Tensor<T, N> ret(x.shape());
		for (const auto [i, item] : enumerate(x))
			ret[i] = std::asinh(item);
		return ret;
	}
	template<typename T, u32 N>
	beta::Tensor<T> cosh(const beta::Tensor<T, N>& x)
	{
		beta::Tensor<T, N> ret(x.shape());
		for (const auto [i, item] : enumerate(x))
			ret[i] = std::cosh(item);
		return ret;
	}
	template<typename T, u32 N>
	beta::Tensor<T> acosh(const beta::Tensor<T, N>& x)
	{
		beta::Tensor<T, N> ret(x.shape());
		for (const auto [i, item] : enumerate(x))
			ret[i] = std::acosh(item);
		return ret;
	}

	template<typename T , u32 N , u32 M>
	beta::Tensor<T> dot(const beta::Tensor<T, N>& a, const beta::Tensor<T, M>& b)
	{
		auto scalar_mul = [](const auto& tensor, T scalar)->beta::Tensor<T>
		{
			beta::Tensor<T>ret(vec::cast_template(tensor.shape()));
			for (const auto& [i, item] : enumerate(tensor))
				ret[i] = item * scalar;
			return ret;
		};
		if (a.rank() == 1 && a.size()==1)
			return scalar_mul(b,a[0]);
		else if ( b.rank() ==1 && b.size() == 1)
			return scalar_mul(a, b[0]);
		else if (a.rank() == 1 && b.rank() == 1 && a.size() == b.size())
		{
			T prod = inner(a.reshape(smallvec<size_t, 1>{a.size()}),
				b.reshape(smallvec<size_t, 1>{b.size()}));
			beta::Tensor<T> ret({ 1 });
			ret[0] = prod;
			return ret;
		}
		else if (a.rank() == 2 && b.rank() == 2)
		{
			beta::Tensor<T> ret = tensor::zeros<T>(smallvec<size_t>{a.shape()[0], b.shape()[1]});
			mat_mul(a, b, ret);
			return ret;
		}
		else if (b.rank() == 1)
		{
			return tensordot(a, b, std::array{ -1, 0 });
		}
		else
		{
			return tensordot(a, b , std::array{-1 , (i32)b.rank()-2 });
		}
		
	}

	template<typename T, u32 N, u32 M>
	Tensor<T> outer(const Tensor<T, N>& a, const Tensor<T, M>& b)
	{
		Tensor<T, N> A = a.reshape(a.size(), 1);
		Tensor<T, M> B = b.reshape(1, b.size());
		return ops::Multiply(A, B);
	}
};//end math

namespace linalg
{
	template<typename T, u32 N = 7>
	using Tensor = beta::Tensor<T, N>;

	template<typename T>
	using Matrix = Tensor<T, 2>;

	template<typename T>
	using Vector = Tensor<T, 1>;
	
	namespace _internal
	{
		/*applies a function to the last two dims of a tensor that uses
		 *two tensors and the indices that are being used to loop
		 */ 
		template<typename T, u32 N, typename functor>
		void const_matrixloop(
			const Tensor<T, N>& tensor,
			Tensor<T, N>& tofill,
			smallvec<size_t, N>& ind,
			i64 i,
			const functor& F
		)
		{
			if (i >= (i64)tensor.rank() - 2) {
				F(tensor, tofill, ind);
				return;
			}
			for (size_t ii = 0; ii < tensor.shape()[i]; ++ii) {
				ind[i] = ii;
				const_matrixloop(tensor, tofill, ind, i + 1, F);

			}
			ind[i] = 0;
		}
		template<typename T, u32 N, typename functor>
		void matrixloop(
			Tensor<T, N>& tensor,
			Tensor<T, N>& tofill,
			smallvec<size_t, N>& ind,
			i64 i,
			const functor& F
		)
		{
			if (i >= (i64)tensor.rank() - 2) {
				F(tensor, tofill, ind);
				return;
			}
			for (size_t ii = 0; ii < tensor.shape()[i]; ++ii) {
				ind[i] = ii;
				matrixloop(tensor, tofill, ind, i + 1, F);

			}
			ind[i] = 0;
		}
	};

	template<typename T , u32 N>
	Tensor<T,N> det(const Tensor<T, N>& matrix)
	{
		if (matrix.shape()[-1] != matrix.shape()[-2])
			std::cerr << "schneed square matrix/stack of matrices",
			throw std::invalid_argument("");

		auto mdet = [](
			Tensor<T, N>& tensor,
			Tensor<T, N>& in,
			smallvec<size_t, N>& i)
		{

			for (size_t o = 0; o < tensor.shape()[-2]; ++o) { 
			  for (size_t k = o+1; k < tensor.shape()[-1]; ++k) {
				 i[-2] = o,i[-1] = o;
				 size_t offset = tensor.calc_offset(i);
				 T s = tensor[offset];
				 if (s == 0)
					 tensor[offset] = 1e-18, s = 1e-18;

				 i[-2] = k;
				 offset = tensor.calc_offset(i);
				 T f = tensor[offset];
				 T crs = f / s;

				 for (size_t j = 0; j < tensor.shape()[-1]; ++j)
				 {
					 i[-1] = j,i[-2] = o;
					 T m = tensor[tensor.calc_offset(i)];
					 i[-2] = k;
					 tensor[tensor.calc_offset(i)] -= crs * m;
				 }	
			  }
			}
			size_t offset = in.calc_offset(
				tensor.rank()==2 ? smallvec<size_t,N>{0} : vec::slice(i, 0, -2)
			);
			T& prod = in[offset];
			prod = 1.;
			for (size_t o = 0; o < tensor.shape()[-1]; ++o)
			{
				i[-1] = o, i[-2] = o;
				prod *= tensor[tensor.calc_offset(i)];
			}
		};

		Tensor<T, N>AM = matrix.deepcopy();
		Tensor<T, N> det;
		smallvec<size_t, N> indices(matrix.rank());
		if (AM.rank() == 2) {
			det = Tensor<T,N>({1});
			mdet(AM, det, indices);
		}
		else {
			det= Tensor<T, N>(vec::slice(matrix.shape(), 0, -2));
			_internal::matrixloop(AM, det, indices, 0, mdet);
		}
		return det;
	}
	

	template<typename T, u32 N>
	Tensor<T, N> inverse(const Tensor<T, N>& matrix)
	{
		[](const auto& matrix)
		{
		  Tensor<T, N> determinant = linalg::det(matrix);
		  for (const auto item : determinant)
			if (item == 0)
				std::cerr << "Singular Matrix exception", throw std::invalid_argument("");
		}(matrix);

		Tensor<T, N> AM= matrix.deepcopy();
		Tensor<T, N> IM = tensor::identity<T,N>(AM.shape());
		smallvec<size_t, N> indices(AM.rank());

		auto inv = [](Tensor<T, N>& AM,
			Tensor<T, N>& IM, smallvec<size_t, N>& i)
		{
			for (size_t o = 0; o < AM.shape()[-1]; ++o){
				i[-2] = o, i[-1] = o;
				T fscr = 1. / AM[AM.calc_offset(i)];
				for (size_t oo = 0; oo < AM.shape()[-1]; ++oo) {
					i[-1] = oo;
					AM[AM.calc_offset(i)] *= fscr;
					IM[IM.calc_offset(i)] *= fscr;
				}
				for (size_t oo = 0; oo < o; ++oo) {
					i[-2] = oo, i[-1] = o;
					T crs = AM[AM.calc_offset(i)];
					for (size_t ii = 0; ii < AM.shape()[-1]; ++ii) {
						i[-1] = ii, i[-2] = o;
						T amfj = AM[AM.calc_offset(i)],
							imfj = IM[IM.calc_offset(i)];
						i[-2] = oo;
						AM[AM.calc_offset(i)] -= crs * amfj;
						IM[IM.calc_offset(i)] -= crs * imfj;
					}
				}
				for (size_t oo = o + 1; oo < AM.shape()[-1]; ++oo) {
					i[-2] = oo, i[-1] = o;
					T crs = AM[AM.calc_offset(i)];
					for (size_t ii =0; ii < AM.shape()[-1]; ++ii) {
						i[-1] = ii, i[-2] = o;
						T amfj = AM[AM.calc_offset(i)],
							imfj = IM[IM.calc_offset(i)];
						i[-2] = oo;
						AM[AM.calc_offset(i)] -= crs * amfj;
						IM[IM.calc_offset(i)] -= crs * imfj;
					}
				}
			}
		};
		_internal::matrixloop(AM, IM, indices, 0,inv);
		return IM;
	}
};//end linalg