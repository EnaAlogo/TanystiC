#pragma once
#include "init.h"
#include "SharedPtr.hpp"



template<typename dtype>
class Variable : public  VarialbeInterface
{
public:
	using value_type = dtype;
	using pointer = dtype*;
	using const_pointer = const dtype*;
	using reference = dtype&;
	using const_reference = const dtype&;

	Variable(SharedPtr<value_type[]>& buffer , bool trainable = 1)
		:buffer_( std::move(buffer) ), trainable(trainable) {};

	
	const Tensor<dtype, N>& get()
	{
		return buffer_;
	}

private:
	SharedPtr<value_type[]> buffer_;
	bool trainable;
};

template<typename dtype>
class Module
{
public:
	using value_type = dtype;
	using reference = value_type&;

	

	
protected:


private:
	std::vector<Variable<value_type>> variables;

};