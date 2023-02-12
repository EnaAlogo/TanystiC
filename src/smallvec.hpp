#pragma once
#include "init.h"
#include "Zip.hpp"



template<typename T, u16 N >
class smallvec
{
public:
    using value_type = T;
    using reference = T&;
    using iterator = std::array<T, N>::iterator;
    using const_iterator = std::array<T, N>::const_iterator;
    using const_reference = const T&;

    constexpr smallvec()
        :size_(0), buff({ 0 }) {};

    constexpr smallvec(size_t size)
        :size_(size), buff({ }) 
    {
        assert(size <= N);
    };

    template<typename ...args,
        typename = std::enable_if_t<sizeof...(args) != 1>>
    constexpr smallvec(args&&...dims)
        :buff( std::forward<args>(dims)... ),
        size_(sizeof...(args)) {};

    constexpr smallvec(const smallvec& other)
        : size_(other.size_), buff(other.buff)
    {

    };

    constexpr smallvec(std::initializer_list<T> l)
        :size_(l.size()) , buff({})
    {
        for (auto [me, list] : zip(*this, l))
            me = list;
    };

    constexpr smallvec(smallvec&& other) noexcept
        :size_(other.size_), buff(std::move(other.buff))
    {
        other.size_ = 0;
    }

 

    constexpr smallvec<T, N>& operator =(const smallvec<T, N>& other) = default;

    constexpr smallvec<T, N>& operator =(smallvec<T, N>&& other) = default;

    constexpr inline const i16 size() const { return size_; };

    constexpr inline bool empty() const { return size_ == 0; }

    constexpr inline const i16 capacity() const { return N; }

    constexpr reference operator[](i64 offset)
    {
        offset = offset >= 0 ? offset : offset + size_;
        assert (offset < size_ &&"subscript out of range");
        return buff[offset];
    }
    constexpr const_reference operator[](i64 offset) const
    {
        offset = offset >= 0 ? offset : offset + size_;
        assert(offset < size_ && "subscript out of range");
        return buff[offset];
    }

    constexpr bool operator ==(const smallvec& other) const
    {
        if (size_ != other.size_)
            return false;
        for (const auto& [a, b] : zip(*this, other))
            if (a != b)
                return false;
        return true;
    }
    constexpr bool operator !=(const smallvec& other) const
    {
        return !(*this == other);
    }

    constexpr void prepend(const T& element)
    {
        ++size_;
        assert(size_ <= N && "cannot expand more");
        std::move(buff.begin(), buff.begin() + size_, buff.begin() + 1);
        buff[0] = element;
    }
    constexpr void prepend(T&& element)
    {
        ++size_;
        assert(size_ <= N && "cannot expand more");
        std::move(buff.begin(), buff.begin() + size_, buff.begin() + 1);
        buff[0] = std::move(element);
    }
    constexpr void prepend(const smallvec<T, N>& left)
    {
        if (size() + left.size() > capacity() || size() + left.size() > left.capacity())
            throw std::invalid_argument("cannot be concatenated exceeds size");
        std::move(buff.begin(), buff.begin() + size_, buff.begin() + left.size());
        std::copy(left.buff.begin(), left.buff.begin() + left.size(), buff.begin());
        size_ += left.size();
    }
    constexpr void append(const T& element)
    {
        if (size_ == N )
            throw std::out_of_range("cannot grow more");
        size_++;
        buff[size_ - 1] = element;
    }
    constexpr void append(T&& element)
    {
        if (size_ == N)
            throw std::out_of_range("cannot grow more");
        size_++;
        buff[size_ - 1] =  std::move(element);
    }
   
    constexpr void append(const smallvec<T, N>& left) 
    {
        if (size() + left.size() > capacity() || size() + left.size() > left.capacity())
            throw std::invalid_argument("cannot be concatenated exceeds size");

        for (T el : left)
            append(el);
    }
    constexpr void resize(i32 size)
    {
        assert(size <= N && "cannot expand more");
        size_ = size;
    }

    template<typename ...args>
    constexpr void emplace_back(args&&...f)
    {
        ++size_;
        assert(size_ <= N && "cannot expand more");
        buff[size_ - 1] = T(std::forward<args>(f)...);
    }

    constexpr void pop()
    {
        assert(size_ > 0 && "cant pop empty vec");
        buff[size_ - 1] = 0;
        size_--;
    }
    constexpr void pop_front()
    {
        assert(size_ > 0 && "cant pop empty vec");
        std::move(buff.begin() +1, buff.begin() + size_, buff.begin());
        size_--;
    }

    constexpr T front() const
    {
        assert(size_ > 0 && "empty vec");
        return buff[0];
    }
    constexpr T back() const
    {
        assert(size_ > 0 && "empty vec");
        return buff[size_ - 1];
    }
    constexpr iterator begin()
    {
        return buff.begin();
    }
    constexpr iterator end()
    {
        return buff.begin() + size_;
    }
    constexpr  const_iterator begin() const
    {
        return buff.begin();
    }
    constexpr const_iterator end() const
    {
        return buff.begin() + size_;
    }

    constexpr bool contains(T smth) const
    {
        for (T el : *this)
            if (el == smth)
                return true;
        return false;
    }
    template<typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    constexpr T prod() const
    {
        T prod_ = 1;
        for (T el : *this)
            prod_ *= el;
        return prod_;
    }
    template<typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    constexpr T prod(const std::function<T(T)>& mod) const
    {
        T prod_ = 1;
        for (T el : *this)
            prod_ *= mod(el);
        return prod_;
    }
    template<typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    constexpr T sum() const
    {
        T prod_ = 0;
        for (T el : *this)
            prod_ += el;
        return prod_;
    }
    template<typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    constexpr T sum(const std::function<T(T)>& mod) const
    {
        T prod_ = 1;
        for (T el : *this)
            prod_ += mod(el);
        return prod_;
    }
    template<typename =std::enable_if_t<std::is_arithmetic_v<T>>>
    constexpr T max() const
    {
        T max = std::numeric_limits<T>::lowest();
        for (const T el : *this)
            max = std::max(max, el);
        return max;
    };
    template<typename= std::enable_if_t<std::is_arithmetic_v<T>>>
    constexpr T argmax() const
    {
        T max = std::numeric_limits<T>::lowest();
        size_t index;
        for (const auto[i , el ] : enumerate(*this) )
            if (el > max) {
                max = el;
                index = i;
            }
        return index;
    };

    template<typename =std::enable_if_t<std::is_arithmetic_v<T>>>
    constexpr T min() const
    {
        T min = std::numeric_limits<T>::max();
        for (const T el : *this)
            min = std::min(min, el);
        return min;
    };
    template<typename =std::enable_if_t<std::is_arithmetic_v<T>>>
    constexpr T argmin() const
    {
        T min = std::numeric_limits<T>::max();
        size_t index;
        for (const auto [i, el] : enumerate(*this))
            if (el < min) {
                min = el;
                index = i;
            }
        return index;
    };

    friend std::ostream& operator << (std::ostream& stream, const smallvec& prnt)
    {
        stream << "( ";
        for (T el : prnt)
            stream << el << " ";
        stream << ')';
        return stream;
    };

    template<typename t, u32 m, u32 y>
    constexpr friend bool operator==(const smallvec<t, m>& left, const smallvec<t, y>& right);

    template<typename t, u32 m, u32 y>
    constexpr friend bool operator!=(const smallvec<t, m>& left, const smallvec<t, y>& right);

    constexpr const std::array<T, N>& c_arr() const
    {
        return buff;
    }

    constexpr std::array<T, N>& c_arr()
    {
        return buff;
    }
private:
    //TODO add heap allocation when overflown option
    //this will also need a new iterator and indexing impl
    std::array<T, N> buff;
    u16 size_;
};




template<typename t ,u32 m, u32 y>
constexpr bool operator==(const smallvec<t, m>& left ,const smallvec<t, y>& right)
{
    if (left.size() != right.size())
        return 0;
    for (const auto& [l, r] : zip(left, right))
        if (l != r)
            return 0;
    return 1;
}
template<typename t, u32 m, u32 y>
constexpr bool operator!=(const smallvec<t, m>& left, const smallvec<t, y>& right)
{
    return !(left == right);
}

namespace vec {

    template<typename T, u32 N>
    constexpr smallvec<T, N> concat(const smallvec<T, N>& right, const smallvec<T, N>& left)
    {
        if (right.size() + left.size() >= right.capacity() || right.size() + left.size() >= left.capacity())
            throw std::invalid_argument("cannot be concatenated exceeds size");

        smallvec<T, N > result = right;
        for (T el : left)
            result.append(el);
        return result;
    }
#if 0
    template<typename T, u32 N , u32 M>
    constexpr smallvec<T, N > concat(const smallvec<T, M>& right, const smallvec<T, N>& left)
    {
        if ( right.size() + left.size() >= left.capacity())
            throw std::invalid_argument("cannot be concatenated exceeds size");

        smallvec<T, N > result = right;
        for (T el : left)
            result.append(el);
        return result;
    }
    template<typename T, u32 N, u32 M>
    constexpr smallvec<T> concat(const smallvec<T, N>& right, const smallvec<T, M>& left)
    {
        
        smallvec<T> result;
        for (T el : right)
            result.append(el);
        for (T el : left)
            result.append(el);
        return result;
    }
#endif
    template<typename T, u32 N>
    constexpr T dot(const smallvec<T, N>& right, const smallvec<T, N>& left)
    {
        T dot = 0;
        for (const auto& [l, r] : zip(left, right))
            dot += l * r;
        return dot;
    }

    template<u32 n, typename...Indices,
    typename = std::enable_if_t<sizeof...(Indices) != 1>>
    constexpr size_t compute_flat_index(const smallvec<size_t , n>& strides, const smallvec<size_t , n>& shape,
        Indices&&..._indices)
    {
        size_t dot = 0;
        smallvec<i64 , n> indices(std::forward<Indices>(_indices)...);
        for (auto [l, r ,i] : zip(shape, strides , indices) ) {
            int index = i >= 0 ? i : l+i;
            assert(index < l && "subscript out of range");
            dot += index * r;
        }
        return dot;
    }
    template<u32 n>
    constexpr size_t compute_flat_index(const smallvec<size_t, n>& strides,
        const smallvec<size_t,n>& shape ,const smallvec<i64, n>&indices)
    {
        size_t dot = 0;
        for (auto [l , r, i] : zip(shape , strides, indices)) {
            i64 index = i >= 0 ? i : l + i;
            assert(index < l && "subscript out of range");
            dot += i * r;
        }
        return dot;
    }
   
    template<u32 n>
    constexpr size_t compute_flat_index(const smallvec<size_t, n>& strides,
        const smallvec<size_t, n>& shape, const smallvec<size_t ,n>& indices)
    {
        size_t dot = 0;
        for (auto [l, r, i] : zip(shape, strides, indices) ) {
            assert(i < l && "subscript out of range");
            dot += i * r;
        }
        return dot;
    }

    template<u32 n>
    constexpr smallvec<size_t, n> compute_strides(const smallvec<size_t, n>& _shape,
        size_t trail_stride = 1)
    {
        i32 size = _shape.size();
        smallvec<size_t, n> strides(size);
        strides[size - 1] = trail_stride;
        for (size_t i = size - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * _shape[i];
        }
        return strides;
    }
    template<typename t , u32 n>
    constexpr smallvec<t, n> slice(const smallvec<t,n>& vec ,i64 start , i64 end )
    {
        start = start >= 0 ? start : start + vec.size();
        end = end >= 0 ? end : vec.size() + end;
        assert(start < end);
        smallvec<t, n> ret;
        for (size_t i = start; i < end; i++)
            ret.append(vec[i]);
        return ret;
    }

    template<typename T, u32 N>
    constexpr void permute(smallvec<T, N>& vec, 
        std::initializer_list<i32> perms , bool do_validate = true)
    {
        if (do_validate) {
            assert(perms.size() > 1 && "empty permutations list");
            auto validate_perms = [&]()
            {
                smallvec<T, N> count(perms.size());
                for (i32 perm : perms)
                    count[perm]++;
                return count.max() == 1;
            };
            assert(validate_perms() && "duplicate indices not allowed");
        }
        smallvec<T, N> temp(vec);
        for (auto [ref, perm] : zip(temp, perms))
            ref = vec[perm];
        vec = std::move(temp);
    }

    template<typename T , u32 N=7>
    constexpr smallvec<T, N> tovec(const range& r)
    {
        smallvec<T, N>ret;
        for (auto i : r)
            ret.append(i);
        return ret;
    };
    template<typename T, u32 M , u32 N = 7>
    constexpr smallvec<T, N> cast_template(const smallvec<T, M>& a)
    {
        smallvec<T, N> ret;
        for (const T& item : a)
            ret.append(item);
        return ret;
    }
    template<typename T , u32 N , u32 M>
    constexpr u8 bit_intersect(const smallvec<T, N>& left, const smallvec<T, M>& right)
    {
        if (left.size() != right.size())return 0;
        u8 bits = 0;
        for (const auto [i, tup] : enumerate(zip(left, right))) {
            const auto& [l, r] = tup;
            if (l == r)
                bits |= (1 << i);
        }
        return bits;
    }
    template<typename T , u32 N , u32 M >
    constexpr bool any_common(const smallvec<T, N>& left, const smallvec<T, M>& right)
    {
        return bit_intersect(left, right) != 0;
    }

};