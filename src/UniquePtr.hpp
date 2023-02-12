#pragma once

template<typename T>
class UniquePtr
{
public:
    using value_type = T;
    using pointer = value_type*;
    using reference = value_type&;
    using const_reference = const T&;
    using const_pointer = const T*;

    constexpr UniquePtr() noexcept {};

    constexpr explicit UniquePtr(T* obj)
        :ptr(obj) {
        obj = nullptr;
    };

    UniquePtr(const UniquePtr& other) = delete;

    UniquePtr(UniquePtr&& other) noexcept
        :ptr(other.ptr)
    {
        other.ptr = nullptr;
    }

    ~UniquePtr()
    {
        delete ptr;
    }
    UniquePtr& operator=(const UniquePtr& other) = delete;
    UniquePtr& operator=(UniquePtr&& other) = delete;

    template<typename P>
    operator UniquePtr<P>()
    {
        UniquePtr<P>casted(static_cast<P*>(ptr));
        ptr = nullptr;
        return casted;
    }

    operator bool()
    {
        return ptr != nullptr;
    }

    const_pointer operator ->()const {
        return ptr;
    }
    pointer operator ->()   {
        return ptr;
    }
    const_reference operator *() const {
        return *ptr;
    }
    reference operator *()  {
        return *ptr;
    }

    void release()
    {
        this->~UniquePtr();
        ptr = nullptr;
    }


private:
    pointer ptr;

};
template<typename T>
class UniquePtr<T[]>
{
public:
    using value_type = T;
    using pointer = value_type*;
    using reference = value_type&;
    using const_reference = const T&;
    using const_pointer = const T*;

    constexpr UniquePtr() noexcept {};

    constexpr explicit UniquePtr(T* obj)
        :ptr(obj) {
        obj = nullptr;
    };

    UniquePtr(const UniquePtr& other) = delete;

    UniquePtr(UniquePtr&& other) noexcept
        :ptr(other.ptr)
    {
        other.ptr = nullptr;
    }

    ~UniquePtr()
    {
        delete[] ptr;
    }

    UniquePtr& operator=(const UniquePtr& other) = delete;
    UniquePtr& operator=(UniquePtr&& other) = delete;

    template<typename P>
    operator UniquePtr<P>()
    {
        UniquePtr<P>casted(static_cast<P*>(ptr));
        ptr = nullptr;
        return casted;
    }

    operator bool()
    {
        return ptr != nullptr;
    }

    const_pointer operator ->()  const {
        return ptr;
    }

    pointer operator ->()   {
        return ptr;
    }

    const_reference operator[](size_t offset) const
    {
        return *(ptr + offset);
    }
    reference operator[](size_t offset)
    {
        return *(ptr + offset);
    }

    const_reference operator *() const {
        return *ptr;
    }
    reference operator *()  {
        return *ptr;
    }

    void release()
    {
        this->~UniquePtr();
        ptr = nullptr;
    }
    

private:
    pointer ptr;

};
template<typename a , typename...args>
UniquePtr<a> make_owner(args&&...fields)
{
    return UniquePtr<a>(new a(std::forward<args>(fields)...));
}
#if 0
template<typename a>
UniquePtr<a[]> make_owner(size_t size)
{
    return UniquePtr<a[]>(new a[size]);
}
#endif