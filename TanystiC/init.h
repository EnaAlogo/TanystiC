#pragma once

#include <array>
#include <numeric>
#include <vector>
#include <type_traits>
#include <assert.h>
#include <unordered_set>
#include <initializer_list>
#include <string>
#include <functional>
#include <ranges>
#include <unordered_map>
#include <random>
#include <optional>
#include <variant>
#include <limits>
#include <execution>
#include <numbers>

typedef uint32_t u32;
typedef __int64 i64;
typedef int32_t i32;
typedef uint16_t u16;
typedef int16_t i16;
typedef uint8_t u8;
typedef int8_t i8;
typedef float f32;
typedef double f64;
typedef unsigned int uint;
typedef unsigned long ulong;

static int allocations = 0;
static int deallocations = 0;

#if 0
void* operator new[](size_t size)
{
	allocations++;
	return malloc(size);
}
void operator delete[](void* ptr)
{
	deallocations++;
	free(ptr);
}
#endif


namespace beta {
	template<typename T, u32 N = 7 >
	class Tensor;
}

class range;

template<typename T, u16 N = 7>
class smallvec;

