#include "init.h"


namespace preprocess
{
	template<typename T>
	using Tensor = beta::Tensor<T, 7>;
#if 0
	template<typename T>
	Tensor<T> one_hot_encode(const Tensor<T>& goals)
	{
		//will prolly need sort
		assert(goals.rank() == 1 || goals.rank() == 2 && "invalid agrument");
		auto find_unique = [](const auto& tensor) ->size_t
		{
			std::unordered_set<T> unique;
			for (const auto& item : (tensor))
				unique.insert(item);
			return unique.size();
		};
		size_t subtensor_size = find_unique(goals);
		if (goals.rank() == 2) {// where dim1 = batch_Size

		}
	}
#endif

};
