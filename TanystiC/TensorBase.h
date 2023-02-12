#pragma once

#include "init.h"

#include "TensorIterators.hpp"
#include "SharedPtr.hpp"
#include "UniquePtr.hpp"
#include "initializers.hpp"
#include "ScalarOps.hpp"
#include "Tensorfuncs.hpp"

namespace beta
{
	template<typename T>
	class expr;

	
	template< typename Type , u32 N >
	class Tensor
	{
		using self = Tensor<Type, N>;
		using value_type = Type;
		using reference = Type&;
		using const_reference = Type const&;
		using pointer = Type*;
		using const_pointer = Type const*;
		using TensorShape = smallvec<size_t , N>;
		using Strides = smallvec<size_t , N>;
	public:
		using iterator = tensor_iterator<Type, N>;
		using const_iterator = const_tensor_iterator<Type, N>;

		constexpr Tensor()
			:begin_(0) ,shape_() , strides_() , data_() {};

		constexpr Tensor(const TensorShape& shape)
			:shape_(shape), begin_(0), 
			strides_(vec::compute_strides(shape)), data_(new value_type[shape.prod()]) {};

		constexpr Tensor(std::initializer_list<size_t> shape)
			:shape_(shape) , begin_(0),
			strides_(vec::compute_strides(shape_)), data_(new value_type[shape_.prod()]) {};

		constexpr Tensor(const TensorShape& shape, const Strides& strides,
			i64 begin , const SharedPtr<value_type[]>& data)
			:shape_(shape), begin_(begin),
			strides_(strides), data_(data) {};

		constexpr Tensor(const TensorShape& shape,
			i64 begin, const SharedPtr<value_type[]>& data)
			:shape_(shape), begin_(begin),
			strides_(vec::compute_strides(shape)), data_(data) {};

		constexpr Tensor(const Tensor& other)
			:shape_(other.shape_), begin_(other.begin_),
			strides_(other.strides_), data_(other.data_) {};

		constexpr Tensor(Tensor&& other) noexcept
			:shape_( std::move(other.shape_) ), begin_( other.begin_ ),
			strides_( std::move(other.strides_) ), data_( std::move(other.data_) ) {};


		constexpr Tensor& operator=(const Tensor& other)
		{
			if (this != &other)
			{
				shape_ = other.shape_;
				strides_ = other.strides_;
				data_ = other.data_;
				begin_ = other.begin_;
			};
			return *this;
		}
		constexpr Tensor& operator=(Tensor&& other) = default;

		constexpr inline u32 rank() const{ return shape_.size(); };
		constexpr inline const TensorShape& shape() const { return shape_; };
		constexpr inline const Strides& strides() const { return strides_; };
		constexpr inline const size_t size() const { return shape_.prod(); };
		constexpr inline const SharedPtr<value_type[]>& data() const { return data_; };
		constexpr inline const i64 offset() const { return begin_; };

		//constexpr inline SharedPtr<value_type[]>& set_data() { return data_; };
		//constexpr inline  i64& set_offset() { return begin_; };
		//constexpr inline TensorShape& set_shape() { return shape_; };
		//constexpr inline Strides& set_strides() { return strides_; };

		template<typename ...Indices>
		constexpr reference operator () (Indices&&...indices)
		{
			static_assert((std::is_convertible_v<Indices, std::size_t> && ...), "Indices must be convertible to size_t");
			assert(sizeof...(Indices) == rank() && "indices len != rank");
			smallvec<i64, N> indices_( std::forward<Indices>(indices)... );
			size_t index = calc_offset(indices_);
			return data_[begin_+index];
		};

		template<typename ...Indices>
		constexpr const_reference operator () (Indices&&...indices) const
		{
			static_assert((std::is_convertible_v<Indices, std::size_t> && ...), "Indices must be convertible to size_t");
			assert(sizeof...(Indices) == rank() && "indices len != rank");
			smallvec<i64, N> indices_( std::forward<Indices>(indices)... );
			size_t index = calc_offset(indices_);
			return data_[begin_+index];
		};

		template<typename ...Indices>
		constexpr const Tensor<value_type> subtensor(Indices&&...indices) const
		{
			//static_assert((std::is_convertible_v<Indices, std::size_t> && ...), "Indices must be convertible to size_t");
			assert(sizeof...(Indices) < rank() && "subscript out of range");
			smallvec<i64, N> indices_( std::forward<Indices>(indices)... );
			size_t index = calc_offset(indices_);
			return Tensor<value_type>( vec::slice(shape_ ,sizeof...(Indices), rank()), index, data_);
		};
		template<typename ...Indices>
		constexpr Tensor<value_type> subtensor(Indices&&...indices) 
		{
			static_assert((std::is_convertible_v<Indices, std::size_t> && ...), "Indices must be convertible to size_t");
			assert(sizeof...(Indices) < rank() && "subscript out of range");
			smallvec<i64, N> indices_(std::forward<Indices>(indices)... );
			size_t index = calc_offset(indices_);
			return Tensor<value_type>(vec::slice(shape_, sizeof...(Indices), rank()), index, data_);
		};

		constexpr Tensor T() const
		{
			return ops::transpose(*this);
		}
		constexpr Tensor deepcopy() const
		{
			Tensor copy(shape_);
#if MT
			enumerate it(*this);
			size_t i = 0;
			std::for_each(std::execution::par, begin(), end(),
				[&copy, &i](auto tup)
				{
					//auto [i, item] = tup;
					copy[i++] = tup;
				}
			);
#else
			for (const auto& [i, el] : enumerate(*this))
				copy[i] = el;
#endif
			return copy;
		}

		virtual constexpr size_t calc_offset(const smallvec<i64, N>& indices) const
		{
			return vec::compute_flat_index(
				strides_, shape_, indices
			);
		};

		virtual constexpr size_t calc_offset(const smallvec<size_t, N>& indices) const
		{
			return vec::compute_flat_index(
				strides_, shape_, indices
			);
		};

		template<u32 M>
		constexpr void deepcopy(Tensor<Type , M>& copy) const
		{
			assert(copy.isContiguous());
#if MT
			enumerate it(*this);
			size_t i = 0;
			std::for_each(std::execution::par,begin(), end(),
				[&copy , &i](auto tup)
				{
					//auto [i, item] = tup;
			        copy[i++] = tup;
				}
			);
#else
			for (const auto& [i, el] : enumerate(*this))
				copy[i] = el;
#endif
		}

		constexpr inline bool isContiguous() const
		{
			for (i32 i = 1; i < (i32)rank(); ++i)
				if (strides_[i] > strides_[i - 1])
					return false;
			return strides_.back() == 1;
		};

		template<u32 M>
		constexpr Tensor<Type , M > reshape(const smallvec<size_t , M>&new_shape) const
		{
			assert(new_shape.prod() == shape_.prod() && "sizes dont match cant reshape");

			if (!isContiguous())
			{
				Tensor<Type , M> copy(new_shape);
				deepcopy(copy);
				return copy;
			}
			return Tensor<Type , M>(new_shape, vec::compute_strides(new_shape), begin_, data_);
		};

		constexpr self reshape(std::initializer_list<size_t> shape) const
		{
			smallvec<size_t> new_shape(shape);
			assert(new_shape.prod() == shape_.prod() && "sizes dont match cant reshape");
			
			if (!isContiguous())
			{
				Tensor<Type> copy(new_shape);
				deepcopy(copy);
				return copy;
			}
			return Tensor(new_shape, vec::compute_strides(new_shape), begin_, data_);
		};

		template<typename func>
		self transform(const func& f) const
		{
			self out(shape_);
			for (auto [i, item] : enumerate(*this))
				out[i] = f(item);
			return out;
		};

		template< typename...Args ,typename= std::enable_if_t< (sizeof...(Args) > 1) >>
		constexpr self reshape(Args&&...args) const
		{
			smallvec<size_t> new_shape(std::forward<Args>(args)...);
			assert(new_shape.prod() == shape_.prod() && "sizes dont match cant reshape");

			if (!isContiguous())
			{
				Tensor<Type> copy(new_shape);
				deepcopy(copy);
				return copy;
			}
			return Tensor<Type>(new_shape, vec::compute_strides(new_shape), begin_, data_);
		};

		constexpr iterator begin()
		{
			return iterator(*this, TensorShape(rank()) );
		}
		constexpr iterator end()
		{
			return iterator(*this, TensorShape(rank()), shape_.prod() + (size_t)begin_ );
		}
		constexpr const_iterator begin() const
		{
			return const_iterator(*this, TensorShape(rank()) );
		}
		constexpr const_iterator end() const
		{
			return const_iterator(*this, TensorShape(rank()), shape_.prod() + (size_t)begin_ );
		}
		
		constexpr const_reference operator [](size_t offset) const
		{
			return data_[begin_ + offset];
		};
		constexpr reference operator[](size_t offset)
		{
			return data_[begin_ + offset];
		};

		constexpr value_type max() const
		{
			value_type max = std::numeric_limits<value_type>::lowest();
			for (const auto item : *this)
				max = std::max(item, max);
			return max;
		}
		constexpr size_t argmax() const
		{
			size_t at = 0;
			value_type max = std::numeric_limits<value_type>::lowest();
			for (const auto [i, item] : enumerate(*this)) 
				if (item > max)
				{
					at = i;
					max = item;
				}
			return at;
		}
		constexpr value_type min() const
		{
			value_type max = std::numeric_limits<value_type>::max();
			for (const auto item : *this)
				max = std::min(item, max);
			return max;
		};
		constexpr size_t argmin() const
		{
			size_t at = -1;
			value_type max = std::numeric_limits<value_type>::max();
			for (const auto [i, item] : enumerate(*this))
				if (item < max)
				{
					at = i;
					max = item;
				}
			return at;
		}
		constexpr value_type mean() const
		{
			return reduce::mean(*this);
		}

		constexpr self& operator /=(value_type scalar)
		{
			 in_placeScale_operation(_div<value_type>{}, scalar);
			 return *this;
		}
		constexpr self& operator *=(value_type scalar)
		{
			 in_placeScale_operation(_mul<value_type>{}, scalar);
			 return *this;
		}
		constexpr self& operator +=(value_type scalar)
		{
			 in_placeScale_operation(_add<value_type>{}, scalar);
			 return *this;
		}
		constexpr self& operator -=(value_type scalar)
		{
			in_placeScale_operation(_sub<value_type>{}, scalar);
			return *this;
		}
		
		constexpr self operator-() const
		{
			return Unary_operation(_neg<value_type>{});
		}
		constexpr self operator +(value_type scalar) const
		{
			return Scale_operation(_add<value_type>{}, scalar);
		}
		constexpr self operator -(value_type scalar) const
		{
			return Scale_operation(_sub<value_type>{}, scalar);
		}
		constexpr self operator *(value_type scalar) const
		{
			return Scale_operation(_mul<value_type>{}, scalar);
		}
		constexpr self operator /(value_type scalar) const
		{
			return Scale_operation(_div<value_type>{}, scalar);
		}
		
		template<typename functor>
		constexpr self& apply(const functor& filter)
		{
#if MT
			std::for_each(std::execution::par, begin(), end(),
				[&filter](auto& item)
				{
					item = filter(item);
				}
			);
#else
			for (auto& item : *this)
				item = filter(item);
#endif
			return *this;
		}
		self& initialize(initializer<value_type>&& kernel_initializer)
		{
			for (auto& item : *this)
				item = kernel_initializer.getNumber();
			return *this;
		}
		self& initialize( initializer<value_type>& kernel_initializer)
		{
			for (auto& item : *this)
				item = kernel_initializer.getNumber();
			return *this;
		}

#if 0
		template<typename t ,u32 n , u32 m>
		friend constexpr auto operator +(const Tensor<t, n>& a, const Tensor<t, m>& b)
		{
			return ops::Add(a, b);
		}
		template<typename t, u32 n, u32 m>
		friend constexpr auto operator -(const Tensor<t, n>& a, const Tensor<t, m>& b)
		{
			return ops::Subtract(a, b);
		}
		template<typename t, u32 n, u32 m>
		friend constexpr auto operator *(const Tensor<t, n>& a, const Tensor<t, m>& b)
		{
			return ops::Multiply(a, b);
		}
		template<typename t, u32 n, u32 m>
		friend constexpr auto operator /(const Tensor<t, n>& a, const Tensor<t, m>& b)
		{
			return ops::Divde(a, b);
		}
#endif
	private:
		TensorShape shape_;
		Strides strides_;
		i64 begin_;
		SharedPtr<value_type[]> data_;
		

		constexpr void in_placeScale_operation(const operation<value_type>& op,
			value_type scalar)
		{
			for (auto& item : *this)
				item = op(item, scalar);
		}
		constexpr Tensor Scale_operation(const operation<value_type>& op,
			value_type scalar) const
		{
			Tensor res(shape_);
#if MT
			enumerate it(*this);
			size_t i = 0;
			std::for_each(std::execution::par, begin(), end(),
				[&res ,&i , scalar](auto tup)
				{
					//auto [i, item] = tup;
			        res[i++] = op(tup, scalar);
				}
			);
#else
			for (auto[ i , item ] : enumerate(*this) )
				res[i] = op(item, scalar);
#endif
			return res;
		}
		constexpr Tensor Unary_operation(const unary<value_type>& op) const
		{
			Tensor res(shape_);
#if MT
			enumerate it(*this);
			size_t i = 0;
			std::for_each(std::execution::par, begin(), end(),
				[&res, &i](auto tup)
				{
					//auto [i, item] = tup;
					res[i++] = op(tup);
				}
			);
#else
			for (auto [i, item] : enumerate(*this))
				res[i] = op(item);
#endif
			return res;
		}
	};




	template<typename T>
	class expr
	{

	public:
		explicit constexpr expr(const std::function < Tensor<T>() >& f)
			:expression(f) , check(tensor) {};
		explicit constexpr expr(const std::function < T() >& f)
			:expression(f) , check(scalar) {};

		constexpr operator Tensor<T>()
		{
			if (check != tensor)
				throw std::bad_cast();
			return expression();
		}
		constexpr operator T ()
		{
			if (check != scalar)
				throw std::bad_cast();
			return expression();
		}

	private:
		static enum ret_type
		{
			tensor,
			scalar
		};
		std::variant < std::function < Tensor<T>() >, std::function < T() > > expression;
		ret_type check;
		
	};

	
};

