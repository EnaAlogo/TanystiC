#pragma once
#include "init.h"



template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
struct operation
{
	virtual T operator () (T left, T right) const = 0;
};
template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
struct _add : public operation<T>
{
	constexpr T operator () (T left, T right) const override
	{
		return left + right;
	};
};
template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
struct _sub : public operation<T>
{
	constexpr T operator () (T left, T right) const override
	{
		return left - right;
	};
};
template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
struct _mul : public operation<T>
{
	constexpr T operator () (T left, T right) const override
	{
		return left * right;
	};
};
template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
struct _div : public operation<T>
{
	constexpr T operator () (T left, T right) const override
	{
		return left / right;
	};
};
template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
struct _mod : public operation<T>
{
	constexpr T operator () (T left, T right) const  override
	{
		return left % right;
	};
};
template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
struct _xor :public operation<T>
{
	constexpr T operator () (T left, T right) const  override
	{
		return left ^ right;
	};
};
template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
struct _or :public operation<T>
{
	constexpr T operator () (T left, T right) const  override
	{
		return left | right;
	};
};
template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
struct _and :public operation<T>
{
	constexpr T operator () (T left, T right) const  override
	{
		return left & right;
	};
};


template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
struct unary
{
	virtual T operator () (T num) const = 0;
};
template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
struct _neg :public unary<T>
{
	constexpr T operator () (T num) const override
	{
		return -num;
	};
};
template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
struct _flip :public unary<T>
{
	constexpr T operator () (T num) const override
	{
		return ~num;
	};
};
template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
struct _not :public unary<T>
{
	constexpr T operator () (T num) const override
	{
		return !num;
	};
};

