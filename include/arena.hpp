#pragma once
#include <cstddef>
#include <cstdint>

struct Arena {
    std::byte* base{};
    std::size_t capacity;
    std::size_t offset;

    void init(void* memory, std::size_t size);
    void* alloc(std::size_t size);
    void reset();
};
