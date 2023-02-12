#pragma once

#include "init.h"

#include "smallvec.hpp"


///////////////////////////////////////////////////////////////////////////////////

template<typename T , u32 n>
class const_tensor_iterator
{

public:
	using iterator_category = std::forward_iterator_tag;
	using difference_type = std::ptrdiff_t;
	using value_type = T;
	using const_pointer = const value_type*;
	using const_reference = value_type const&;
	using self = const_tensor_iterator;
	using pointer = value_type*;
	using reference = const_reference;
	using indices_type = smallvec<size_t, n>;

	constexpr const_tensor_iterator() {};

	constexpr const_tensor_iterator(const beta::Tensor<T, n>& tensor, const smallvec<size_t, n>& indices)
		:tensor(tensor), indices(indices), offset(tensor.calc_offset(indices)) {};

	constexpr const_tensor_iterator(const beta::Tensor<T, n>& tensor, const smallvec<size_t, n>& indices, size_t offset)
		:tensor(tensor), indices(indices), offset(offset) {};

	constexpr const_tensor_iterator(const const_tensor_iterator& other)
		:tensor(other.tensor), indices(other.indices), offset(other.offset) {};
	constexpr const_tensor_iterator(const_tensor_iterator&& other) noexcept
		:tensor(std::move(other.tensor)), indices(std::move(other.indices)), 
		offset(std::move(other.offset)) {};

	constexpr const_tensor_iterator& operator=(const const_tensor_iterator& other) {
		if (this != &other)
		{
			tensor = other.tensor;
			indices = other.indices;
			offset = other.offset;
		}
		return *this;
	}
	constexpr pointer operator ->() const
	{
		return &tensor[offset];
	}

	constexpr self& operator ++()
	{
		for (int i = tensor.rank() - 1; i >= 0; --i)
			if (indices[i] == tensor.shape()[i] - 1) {
				indices[i] = 0;
				if (i == 0) {
					offset = tensor.end().offset;
					return *this;
				}
			}
			else {
				indices[i]++;
				break;
			}
		offset = tensor.calc_offset(indices);
			//vec::dot(tensor.strides(), indices);
		return *this;
	}
	constexpr self& operator ++(int)
	{
		auto tmp = *this;
		++* this;
		return tmp;
	}

	constexpr bool operator ==(const const_tensor_iterator& other) const
	{
		return offset == other.offset;
	}
	constexpr bool operator!=(const const_tensor_iterator& other) const
	{
		return offset != other.offset;
	}
	constexpr const_reference operator *() const
	{
		return tensor[offset];
	}
	constexpr const smallvec<size_t, n>& Indices() const
	{
		return indices;
	};

private:
	const beta::Tensor<T, n>& tensor;
	smallvec<size_t, n> indices;
	size_t offset;
};

template<typename T, u32 n>
class tensor_iterator
{

public:
	using iterator_category = std::forward_iterator_tag;
	using difference_type = std::ptrdiff_t;
	using value_type = T;
	using const_pointer = const value_type*;
	using const_reference = value_type const&;
	using self = tensor_iterator;
	using pointer = value_type*;
	using reference = value_type&;
	using indices_type = smallvec<size_t, n>;

	constexpr tensor_iterator() {};

	constexpr tensor_iterator(beta::Tensor<T, n>& tensor, const smallvec<size_t, n>& indices)
		:tensor(tensor), indices(indices), offset(tensor.calc_offset(indices)) {};

	constexpr tensor_iterator(beta::Tensor<T, n>& tensor, const smallvec<size_t, n>& indices, size_t offset )
		:tensor(tensor), indices(indices) , offset(offset) {}; 

	constexpr tensor_iterator(const tensor_iterator& other)
		:tensor(other.tensor), indices(other.indices),offset(other.offset) {};

	constexpr tensor_iterator(tensor_iterator&& other) noexcept
		:tensor(std::move(other.tensor)), indices(std::move(other.indices)),
		offset(std::move(other.offset)) {}

	constexpr tensor_iterator& operator=(const tensor_iterator& other)
	{
		if (this != &other)
		{
			tensor = other.tensor;
			indices = other.indices;
			offset = other.offset;
		}
		return *this;
	}
	

	constexpr self& operator ++()
	{
		for (int i = tensor.rank() - 1; i >= 0; --i)
			if (indices[i] == tensor.shape()[i] - 1) {
				indices[i] = 0;
				if (i == 0) {
					offset = tensor.end().offset;
					return *this;
				}
			}
			else {
				indices[i]++;
				break;
			}
		offset = tensor.calc_offset(indices);
			//vec::dot(tensor.strides(), indices);
		return *this;
	}
	constexpr self& operator ++(int)
	{
		auto tmp = *this;
		++* this;
		return tmp;
	}

	constexpr bool operator ==(const tensor_iterator& other) const
	{
		return offset == other.offset;
	}
	constexpr bool operator!=(const tensor_iterator& other) const
	{
		return offset != other.offset;
	}
	constexpr reference operator *() const
	{	
		return tensor[offset];
	}
	
	constexpr const smallvec<size_t, n>& Indices() const
	{
		return indices;
	};

private:
	beta::Tensor<T, n>& tensor;
	smallvec<size_t, n> indices;
	size_t offset;
};

template<typename tensorIt>
class with_indices
{
public:
	using iterator_category = std::forward_iterator_tag;
	using difference_type = std::ptrdiff_t;
	using value_type = tensorIt::value_type;
	using const_pointer = const value_type*;
	using const_reference = value_type const&;
	using self = with_indices;
	using pointer = value_type*;
	using reference = value_type&;

	constexpr with_indices(const tensorIt& iter)
		:iterator(iter) {};
	constexpr with_indices(const with_indices& other)
		:iterator(other.iterator) {};
	
	self& operator++()
	{
		++iterator;
		return *this;
	}
	self& operator++(int)
	{
		auto tmp = *this;
		++* this;
	}

	constexpr bool operator ==(const with_indices& other) const
	{
		return iterator == other.iterator;
	}
	constexpr bool operator!=(const with_indices& other) const
	{
		return iterator != other.iterator;
	}

	constexpr decltype(auto) operator *() const
	{
		typename tensorIt::reference  ref = *iterator;
		const typename  tensorIt::indices_type& iref = iterator.Indices();
		std::pair< const typename  tensorIt::indices_type&, typename tensorIt::reference>
			ret = { iref , ref };
		return ret;
	}

	

private:
	tensorIt iterator;
};

template<typename T , u32 N > 
class ConstIndices
{
public:
	constexpr ConstIndices(const beta::Tensor<T, N>& tensor)
		:tensor(tensor) {};

	constexpr auto begin()
	{
		return with_indices(tensor.begin());
	}
	constexpr auto end()
	{
		return with_indices(tensor.end());
	}
private:
	const beta::Tensor<T, N>& tensor;
};

template<typename T, u32 N >
class Indices
{
public:
	constexpr Indices(beta::Tensor<T, N>& tensor)
		:tensor(tensor) {};

	constexpr auto begin()
	{
		return with_indices(tensor.begin());
	}
	constexpr auto end()
	{
		return with_indices(tensor.end());
	}
private:
	beta::Tensor<T, N>& tensor;
};

template<typename T , u32 N>
auto tEnumerate(beta::Tensor<T, N>& tensor) -> Indices<T,N>
{
	return Indices(tensor);
}
template<typename T , u32 N>
auto tEnumerate(const beta::Tensor<T, N>& tensor) -> ConstIndices< T ,N >
{
	return ConstIndices(tensor);
}


