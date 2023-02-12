#pragma once
#include "init.h"
#include "SharedPtr.hpp"

static std::random_device device;
static std::mt19937 generator;

template<typename value_type>
class initializer
{
public:
    virtual inline value_type getNumber()  = 0;
    virtual ~initializer() {};
};


template<typename value_type, typename = std::enable_if_t<std::is_floating_point_v<value_type>>>
class Random_normal : public initializer<value_type>
{
    
    std::normal_distribution<value_type> dist;

public:
    Random_normal(value_type mean = 0., value_type stddev = .05)
    {
        dist = std::normal_distribution<value_type>{ mean , stddev };
    };

    inline value_type getNumber() override
    {
        return dist(generator);
    }


};
template<typename value_type, typename = std::enable_if_t<std::is_floating_point_v<value_type>>>
class Uniform : public initializer<value_type>
{
    
    std::uniform_real_distribution<value_type> dist;

public:
    Uniform(value_type minval = -.05, value_type maxval = .05)
    {
        dist = std::uniform_real_distribution<value_type>{ minval , maxval };
    };

    inline value_type getNumber()  override
    {
        return dist(generator);
    }
  

};
template<typename value_type>
class Zeros : public initializer<value_type>
{
public:
    inline value_type getNumber()  override {
        return 0;
   }
};
template<typename value_type>
class Ones : public initializer<value_type>
{

public:
    inline value_type getNumber()  override {
        return 1;
    }
};
template<typename value_type, typename = std::enable_if_t<
std::is_arithmetic_v<value_type>>>
class anyNumber : public initializer<value_type>
{

    value_type num;
public:
    anyNumber(value_type num)
        :num(num) {};

    inline value_type getNumber()  override {
        return num;
    }
};
template<typename value_type, typename = std::enable_if_t<
    std::is_arithmetic_v<value_type>>>
class Iota : public initializer<value_type>
{

    value_type from, stride;
public:
    Iota(value_type from = 0, value_type stride = 1)
        :from(from) , stride(stride) {};

    inline value_type getNumber()  override {
        value_type num = from;
        from += stride;
        return num;
    }
};
namespace initializers {
    template<typename value_type>
    static initializer<value_type>* get(const std::string& name)
    {
        if (name == "")
            return nullptr;
        if (name == "random_normal")
        {
            return  new Random_normal<value_type>();
        }
        else if (name == "uniform")
        {
            return  new Uniform<value_type>();
        }
        else if (name == "ones")
        {
            return   new Ones<value_type>();
        }
        else if (name == "zeros")
        {
            return  new Zeros<value_type>();
        }
        throw std::exception("wrong name");
    }
};
#if 0
namespace tensor {

    template<typename T, typename...args,typename= std::enable_if_t<(sizeof...(args)) != 1 >>
    Tensor<T, sizeof...(args)> ones(args&&...shape) {
        Tensor<T, sizeof...(args)> res(shape...);
        for (auto& el : res)
            el = 1;
        return res;
    }
    template<typename T, u32 N>
    Tensor<T, N> ones(const std::array<size_t, N>& shape) {
        Tensor<T, N> res(shape);
        for (auto& el : res)
            el = 1;
        return res;
    }
    template<typename T, typename...args, typename = std::enable_if_t<(sizeof...(args)) != 1 >>
    Tensor<T, sizeof...(args)> zeros(args&&...shape)
    {
        Tensor<T, sizeof...(args)> res(shape);
        for (auto& el : res)
            el = 0;
        return res;
    }
    template<typename T, u32 N>
    Tensor<T, N> zeros(const std::array<size_t, N>& shape)
    {
        Tensor<T, N> res(shape);
        for (auto& el : res)
            el = 0;
        return res;
    }
}
#endif

namespace tensor {
    template<typename T , u32 N>
    using Tensor = beta::Tensor<T, N>;


    template<typename T, u32 N>
    Tensor<T, N> zeros(const smallvec<size_t, N>& shape)
    {
        size_t blocksize = shape.prod();
        SharedPtr<T[]> data(new T[blocksize]{});
        return Tensor<T, N>(
            shape, vec::compute_strides(shape), 0, data
            );
    };

    template<typename T , u32 N = 2 >
    Tensor<T, N> identity(size_t n)
    {
        Tensor<T, N> identity_matrix = tensor::zeros<T,N>({ n,n });
        for (uint i = 0; i < n; i++)
                identity_matrix(i, i) = 1;
        return identity_matrix;
    }
    template<typename T, u32 N  >
    Tensor<T, N> identity(const smallvec<size_t,N>& shape)
    {
        if (shape[-2] != shape[-1])
            std::cerr << "must be tensor of shape (...,m,m), exception not square",
            throw std::invalid_argument("");

        auto loop = [](
            Tensor<T, N>& tensor,
            smallvec<size_t, N>& ind,
            i64 i,
            auto&& call)
        {
            if (i >= tensor.rank() - 2) {
                for(size_t  o =0 ; o < tensor.shape()[-2] ; o++)
                {
                    ind[-1] = o, ind[-2] = o;
                    tensor[tensor.calc_offset(ind)] = 1;
                }
                ind[-1] = 0, ind[-2] = 0;
                return;
            }
            for (size_t ii = 0; ii < tensor.shape()[i]; ++ii) {
                ind[i] = ii;
                call(tensor, ind, i + 1, call);

            }
            ind[i] = 0;
        };
        Tensor<T, N> im = tensor::zeros<T>(shape);
        smallvec<size_t, N> i(shape.size());
        loop(im, i, 0, loop);
        return im;
    }

    template<typename T, u32 N>
    Tensor<T, N> ones(const smallvec<size_t, N>& shape)
    {
        
        size_t blocksize = shape.prod();
        SharedPtr<T[]> data( new T[blocksize] );
        for (i32 i = 0; i < blocksize; ++i)
            data[i] = 1;
        smallvec<size_t, N> strides = vec::compute_strides(shape);
        
        return Tensor<T, N>(shape, strides, 0, data);

    };
    

    template<typename t , u32 N ,
    typename = std::enable_if_t<std::is_arithmetic_v<t>>>
    Tensor<t, N> uniform(const smallvec<size_t, N>& shape ,t min = -.05 ,t max = .05)
    {
        Uniform<t> unifrm(min, max);
        size_t blocksize = shape.prod();
        SharedPtr<t[]> data(new t[blocksize] );
        for (i32 i = 0; i < blocksize; ++i)
            data[i] = unifrm.getNumber();
        return Tensor<t, N>(
            shape, vec::compute_strides(shape), 0, data
            );
    }
  
    template<typename t ,u32 N,
    typename = std::enable_if_t<std::is_floating_point_v<t>>>
    Tensor<t, N> random_normal(const smallvec<size_t, N>& shape, t mean = 0, t stddev = .05)
    {
        Random_normal<t> rn(mean, stddev);
        size_t blocksize = shape.prod();
        SharedPtr<t[]> data(new t[blocksize] );
        for (i32 i = 0; i < blocksize; ++i)
            data[i] = rn.getNumber();
        return Tensor<t, N>(
            shape, vec::compute_strides(shape), 0, data
            );
    }
 
    template<typename T , u32 N = 1 >
    Tensor<T,N> arange(T min , T max ,T step =1)
    {
        if (step == 0)
          std::cerr<<"step cant be 0",  throw std::invalid_argument("");

        size_t blocksize = std::floor( (f32)(max - min) / (f32)step) ;

        SharedPtr<T[]> data(new T[blocksize] );
        size_t ptr = 0;
        for (T i = min ; i < max ; i+=step )
            data[ptr++] = i;

        smallvec<size_t, N> shape = { blocksize };
        return Tensor<T, N>(
            shape, vec::compute_strides(shape), 0, data
            );
    }
    
    
    
};//end tensor