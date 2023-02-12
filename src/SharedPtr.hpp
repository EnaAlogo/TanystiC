#pragma once
#include "init.h"



template<class T>
class SharedPtr {

public:
	using value_type = T;
	using pointer = value_type*;
	using reference = value_type&;
	using const_reference = const value_type*;
	using const_pointer = const value_type&;


	constexpr SharedPtr() noexcept
		:item(nullptr), ref(nullptr) {};


	constexpr explicit SharedPtr(T* obj)
		:item(obj), ref(new u16{ 1 }) {
		
	};

	~SharedPtr()
	{
		dec_count();
	}

	SharedPtr(const SharedPtr& copy)
		: item(copy.item), ref(copy.ref) {
		if (ref)
			(*ref)++;
	}
	SharedPtr(SharedPtr&& copy) noexcept
		: item(copy.item), ref(copy.ref) {
		if (ref)
			(*ref)++;
		copy.release();
	}
	constexpr explicit SharedPtr(T* obj, u16* ref)
		:item(obj), ref(ref)
	{
	};

	template<typename P>
	operator SharedPtr<P>()
	{
		SharedPtr<P> result(static_cast<P*>(item) , ref);
		(*ref)++;
		release();
		return result;
	}

	operator bool()
	{
		return item != nullptr;
	}

	SharedPtr& operator = (const SharedPtr& copy)
	{
		if (this != &copy) {

			release();
			item = copy.item;
			ref = copy.ref;
			(*ref)++;

		}
		return *this;
	}
	SharedPtr& operator = (SharedPtr&& copy) noexcept
	{
		if (this != &copy) {

			release();
			item = copy.item;
			ref = copy.ref;
			(*ref)++;
			copy.release();

		}
		return *this;
	}

	void release() {
		dec_count();
		item = nullptr;
		ref = nullptr;
	}

	const_pointer operator -> () const {
		return item;
	}

    pointer operator -> ()  {
		return item;
	}

	const_reference operator * () const {
		return *item;
	}

	reference operator * ()  {
		return *item;
	}

private:
	pointer item;
	u16* ref;

	void dec_count()
	{
		if (ref && !--* ref) {
			delete item;
			delete ref;
		}
	}
};

template<class T>
class SharedPtr<T[]> {
	
public:
	using value_type = T;
	using pointer = value_type*;
	using reference = value_type&;
	using const_reference = const reference;
	using const_pointer = const pointer;


	constexpr SharedPtr() noexcept
		:item(nullptr), ref(nullptr) {};


	constexpr explicit SharedPtr(T* obj)
		:item(obj), ref(new u16{ 1 }) 
	{
	};

	constexpr explicit SharedPtr(T* obj , u16* ref)
		:item(obj), ref(ref)
	{
	};

	~SharedPtr()
	{
		dec_count();
	}

	SharedPtr(const SharedPtr& copy)
		: item(copy.item), ref(copy.ref) {
		if (ref)
			(*ref)++;
	}

	SharedPtr(SharedPtr&& copy) noexcept
		: item(copy.item), ref(copy.ref) {
		if (ref)
			(*ref)++;
		copy.release();
	}
	
	template<typename P>
	operator SharedPtr<P>()
	{
		SharedPtr<P> result(static_cast<P*>(item), ref);
		(*ref)++;
		release();
		return result;
	}

	operator bool()
	{
		return item != nullptr;
	}

	SharedPtr& operator = (const SharedPtr& copy)
	{
		if (this != &copy) {
	
			release();

			item = copy.item;
			ref = copy.ref;
			(*ref)++;

		}
		return *this;
	}
	SharedPtr& operator = (SharedPtr&& copy) noexcept
	{
		if (this != &copy) {

			release();

			item = copy.item;
			ref = copy.ref;
			(*ref)++;
			copy.release();

		}
		return *this;
	}

	void release() {
		dec_count();
		item = nullptr;
		ref = nullptr;
	}
	
	const_pointer operator -> ()const {
			return item;
	}
	pointer operator -> (){
		return item;
	}

	const_reference operator[](size_t offset) const {
		return *(item + offset);
	}
	reference operator[](size_t offset)  {
		return *(item + offset);
	}

	const_reference operator * ()const {
		return *item;
	}
	reference operator * () {
		return *item;
	}
	
private:
	u16* ref;
	pointer item;

	void dec_count()
	{
		// ref != nullptr && (!ref || !(--(*ref)) )
		if ( ref && !--*ref) {
			delete[] item;
			delete ref;
		}
	}
};


template< class T ,typename ...varargs>
SharedPtr<T> let_pointer(varargs && ...args)
{
	return SharedPtr<T>(new T(std::forward<varargs>(args)...));
}
