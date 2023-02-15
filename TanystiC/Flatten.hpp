#pragma once
#include "init.h"
#include "Layer.hpp"


template<typename T>
class Flatten : public Layer<T>
{
	
	using Tensor = beta::Tensor<T, 7>;
public:
	using value_type = T;

	Flatten(bool isBatch = false)
		:isBatch(isBatch) {};


	Tensor call(const Tensor& inputs , const bool training) override
	{
		smallvec<size_t> out = [this, &inputs]() {
			if (isBatch) {
				smallvec<size_t> flatshape = compute_batched_output_shape(inputs.shape());
				size_t flatsize = flatshape.prod();
				return smallvec<size_t>(inputs.shape()[0], flatsize);
			}
			else {
				return smallvec<size_t>{ 1, inputs.size() };
			}
		}();
		if (training) 
			input_shape = inputs.shape();

		if (inputs.isContiguous())
			return inputs.reshape(out).deepcopy();
		else
		    return inputs.reshape( out );
	};

	Tensor backwards(const Tensor& out_grad , f64 lr) override
	{
		return out_grad.reshape(input_shape);
	}

private:
	smallvec<size_t> input_shape;
	bool isBatch;

	inline smallvec<size_t> compute_batched_output_shape(const smallvec<size_t>&shape) const
	{
		return vec::slice(shape, 1, shape.size());
	}

	
};