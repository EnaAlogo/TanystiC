#pragma once
#include "init.h"

template <typename ... Args, size_t ... Index>
constexpr auto any_match_impl(std::tuple<Args...> const& lhs,
    std::tuple<Args...> const& rhs,
    std::index_sequence<Index...>) -> bool {
    return (... | (std::get<Index>(lhs) == std::get<Index>(rhs)));
}

template <typename ... Args>
constexpr auto any_match(std::tuple<Args...> const& lhs,
    std::tuple<Args...> const& rhs) -> bool {
    return any_match_impl(lhs, rhs, std::index_sequence_for<Args...>{});
}

template <typename... Iterators>
class zip_iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::tuple<typename std::iterator_traits<Iterators>::value_type...>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = std::tuple<typename std::iterator_traits<Iterators>::reference...>;
    using const_reference = value_type const&;
    using const_pointer = const value_type*;
    using tuple_type = std::tuple<Iterators...>;

    constexpr zip_iterator(Iterators... iters) : iterators_(iters...) {}

    constexpr auto operator++() -> zip_iterator& {
        std::apply([](auto & ... args) {
            ((++args), ...);
            }, iterators_);
        return *this;
    }

    constexpr auto operator++(int) -> zip_iterator {
        auto tmp = *this;
        ++* this;
        return tmp;
    }

    constexpr reference operator*()
    {
        return std::apply(
            [](auto&&...args)
            {
                return reference((*args) ...);
            }, iterators_
        );
    }
    constexpr const_reference operator*()   const
    {
        return std::apply(
            [](auto&&...args) 
            {
                return reference((*args) ...);
            }, iterators_
        );
    }

    constexpr pointer operator->() {
        return &iterators_;
    }

    constexpr const_pointer operator->() const {
        return &iterators_;
    }

    constexpr bool operator==(const zip_iterator& other) const {
        return any_match(iterators_, other.iterators_);
    }


    constexpr  bool operator!=(const zip_iterator& other) const {
        return !(*this == other);
    }

private:
    std::tuple<Iterators...> iterators_;
};





template <typename... Containers>
class zip_it {

public:
    
    using iterator = zip_iterator<typename Containers::iterator...>;
    using const_iterator = zip_iterator<typename Containers::const_iterator...>;
    template< typename = std::enable_if_t<sizeof...(Containers) != 1 > >
    constexpr zip_it(Containers&... containers) : containers_(containers...) {};

    constexpr zip_it(std::tuple<Containers&...> containers) : containers_(containers) {};

  
    constexpr iterator begin()
    {
        return std::apply(
            [](auto&&...args)
            {
                return iterator(std::begin(args)...);
            }, containers_
        );
    }
  
    constexpr iterator end()
    {
        return std::apply(
            [](auto&&...args)
            {
                return iterator( std::end(args)... );
            }, containers_
        );
    }
   

private:

    std::tuple<Containers&...> containers_;
};

template <typename... Containers>
class const_zip  {


public:
    using iterator = zip_iterator<typename Containers::const_iterator...>;
    using const_iterator = zip_iterator<typename Containers::const_iterator...>;

    template< typename = std::enable_if_t<sizeof...(Containers) != 1 > >
    constexpr const_zip(Containers&... containers) : containers_(containers...) {};

    constexpr const_zip(std::tuple<Containers&...> containers) : containers_(containers) {};

   
    constexpr const_iterator begin()
    {
        return std::apply(
            [](auto&&...args)
            {
                return const_iterator(std::begin(args)...);
            }, containers_
        );
    }

    constexpr const_iterator end()
    {
        return std::apply(
            [](auto&&...args)
            {
                return const_iterator(std::end(args)...);
            }, containers_
        );
    }

private:

    std::tuple<Containers&...> containers_;
};


template <typename... Containers>
zip_it(Containers&...) -> zip_it<Containers...>;


template <typename... Containers >
const_zip(Containers&...) -> const_zip<Containers...>;

template <typename... Containers, typename = std::enable_if_t< (std::is_const_v<Containers> || ...) >>
constexpr const_zip<Containers...> zip(Containers&...c)
{
    return const_zip(c...);
}
template <typename... Containers, typename = std::enable_if_t< !(std::is_const_v<Containers> || ...) >>
constexpr zip_it<Containers...> zip(Containers&...c)
{
    return zip_it(c...);
}

template<typename iterable>
class enumerate
{

public:

    constexpr enumerate(const iterable& object)
        :object(object) {};
    constexpr enumerate(const enumerate& other)
        :object(other.object) {};

    template<typename _iterator>
    class iterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        
        constexpr iterator(const _iterator& it)
            :obj(it) {};
        constexpr iterator(const iterator& other)
            :obj(other.obj), index(other.index) {};

        constexpr  iterator& operator++()
        {
            ++obj;
            index++;
            return *this;
        };
        constexpr iterator& operator++(int)
        {
            auto tmp = *this;
            ++* this;
            return tmp;
        }

        template <typename> struct is_tuple : std::false_type {};
        template <typename ...T> struct is_tuple<std::tuple<T...>> : std::true_type {};
    
        constexpr auto operator*() const
        {
            return std::make_tuple(index, obj.operator*());
        }
      
        constexpr bool operator==(const iterator& other) const
        {
            return obj == other.obj;
        };
        constexpr bool operator!=(const iterator& other) const
        {
            return obj != other.obj;
        };
    private:
        size_t index = 0;
        _iterator obj;
    };

    
    constexpr auto begin()
    {
        return iterator<iterable::iterator>( object.begin() );
    }
    constexpr auto end()
    {
        return iterator<iterable::iterator>( object.end() );
    }
private:
    iterable object;
};

class range
{
public:
    class range_iterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = i64;
        using pointer = i64;
        using reference = i64;
        using const_reference = i64;
        using const_pointer = i64;
        using self = range_iterator;

        constexpr range_iterator(i64 index, i64 step)
            :index(index), step(step) {};

        constexpr range_iterator(const range_iterator& other)
            :index(other.index), step(other.step) {};

        constexpr range_iterator& operator++()
        {
            index += step;
            return *this;
        }
        constexpr value_type operator*() const
        {
            return index;
        }
        constexpr range_iterator& operator++(int)
        {
            self copy = *this;
            index += step;
            return *this;
        }
        constexpr bool operator ==(const range_iterator& other) const
        {
            return step > 0 ? index >= other.index : index <= other.index;
        }
        constexpr bool operator !=(const range_iterator& other) const
        {
            return !(*this == other);
        }
    private:
        i64 index;
        i64 step;
    };

    using iterator = range_iterator;
    using const_iterator = range_iterator;
    using value_type = i64;
    using pointer = i64;
    using reference = i64;

    constexpr range()
        :start(0), end_(0), step(0) {};

    constexpr range(i64 end)
        :start(0), end_(end), step(1) {};

    constexpr range(i64 start, i64 end)
        :start(start), end_(end), step(1) { };

    constexpr range(i64 start, i64 end, i64 step)
        :start(start), end_(end), step(step) {};

    constexpr range(const range& other)
        :start(other.start), end_(other.end_), step(other.step) {};

    constexpr iterator begin() const
    {
        validate_range();
        return iterator(start, step);
    }
    constexpr iterator end() const
    {
        validate_range();
        return iterator(end_, step);
    }
    constexpr bool in(i64 num)const
    {
        return num >= start && num < end_;
    }

    constexpr i64 get_start() const { return start; };
    constexpr i64 get_end() const  { return end_; };
    constexpr i64 get_step() const  { return step; };

private:
    i64 start;
    i64 end_;
    i64 step;

    constexpr void validate_range() const
    {
        assert(step != 0
            && ((start <= end_ && step > 0) ||
                (start >= end_ && step < 0)) && "invalid range");
    }
};