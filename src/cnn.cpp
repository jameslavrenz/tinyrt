#include "cnn.hpp"
#include "ops.hpp"
#include "tensor_factory.hpp"
#include "tensor_access.hpp"
#include <array>
#include <cmath>
#include <cstring>

using namespace Ops;
using namespace TensorFactory;

namespace
{
    uint32_t CalcOutputDim(uint32_t input_dim, int kernel_size, int stride)
    {
        return static_cast<uint32_t>((static_cast<int>(input_dim) - kernel_size) / stride + 1);
    }

    void FlattenNhwc(const Tensor& input, Tensor& output)
    {
        const float* in = tensor_data_f32(const_cast<Tensor&>(input));
        float* out = tensor_data_f32(output);
        std::memcpy(out, in, static_cast<std::size_t>(input.num_elements) * sizeof(float));
    }
}

void Conv2DLayer::forward(const Tensor& input, Tensor& output)
{
    conv.forward(input, output);

    switch (activation)
    {
        case ConvActivationType::None:
            break;
        case ConvActivationType::ReLU:
            ReLU(output, output);
            break;
        case ConvActivationType::Sigmoid:
            Sigmoid(output, output);
            break;
        case ConvActivationType::Tanh:
            Tanh(output, output);
            break;
        case ConvActivationType::LeakyReLU:
            LeakyReLU(output, output, leaky_alpha);
            break;
        case ConvActivationType::ReLU6:
            ReLU6(output, output);
            break;
        case ConvActivationType::Softmax:
            Softmax(output, output);
            break;
    }
}

void MaxPool2DLayer::forward(const Tensor& input, Tensor& output)
{
    const float* in = tensor_data_f32(const_cast<Tensor&>(input));
    float* out = tensor_data_f32(output);

    const uint32_t channels = input.shape[2];
    const uint32_t out_h = output.shape[0];
    const uint32_t out_w = output.shape[1];

    for (uint32_t c = 0; c < channels; ++c)
    {
        for (uint32_t oh = 0; oh < out_h; ++oh)
        {
            for (uint32_t ow = 0; ow < out_w; ++ow)
            {
                float max_val = -INFINITY;
                for (int kh = 0; kh < pool_size; ++kh)
                {
                    for (int kw = 0; kw < pool_size; ++kw)
                    {
                        const uint32_t ih = oh * static_cast<uint32_t>(stride) + static_cast<uint32_t>(kh);
                        const uint32_t iw = ow * static_cast<uint32_t>(stride) + static_cast<uint32_t>(kw);
                        const uint32_t in_idx = index_nhwc(input, ih, iw, c);
                        if (in[in_idx] > max_val)
                            max_val = in[in_idx];
                    }
                }

                const uint32_t out_idx = (oh * out_w + ow) * channels + c;
                out[out_idx] = max_val;
            }
        }
    }
}

CNNNetwork::CNNNetwork(uint32_t num_layers, Arena& arena)
    : blocks(nullptr), num_layers(num_layers), intermediate_outputs(nullptr), arena(arena)
{
    blocks = static_cast<CnnBlock*>(arena.alloc(sizeof(CnnBlock) * num_layers, alignof(CnnBlock)));
    intermediate_outputs = static_cast<Tensor*>(arena.alloc(sizeof(Tensor) * num_layers, alignof(Tensor)));
}

void CNNNetwork::InitConvLayer(uint32_t layer_idx,
                               int kernel_size,
                               int stride,
                               int in_channels,
                               int out_channels,
                               float* weights,
                               float* bias,
                               ConvActivationType activation,
                               float leaky_alpha)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::Conv2D;
    blocks[layer_idx].conv.conv.kernel_size = kernel_size;
    blocks[layer_idx].conv.conv.stride = stride;
    blocks[layer_idx].conv.conv.in_channels = in_channels;
    blocks[layer_idx].conv.conv.out_channels = out_channels;
    blocks[layer_idx].conv.conv.weights = weights;
    blocks[layer_idx].conv.conv.bias = bias;
    blocks[layer_idx].conv.activation = activation;
    blocks[layer_idx].conv.leaky_alpha = leaky_alpha;
}

void CNNNetwork::InitPoolLayer(uint32_t layer_idx, int pool_size, int stride)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::MaxPool2D;
    blocks[layer_idx].pool.pool_size = pool_size;
    blocks[layer_idx].pool.stride = stride;
}

void CNNNetwork::InitFlattenLayer(uint32_t layer_idx)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::Flatten;
}

void CNNNetwork::InitDenseLayer(uint32_t layer_idx,
                                const Tensor& weights,
                                const Tensor& bias,
                                ActivationType activation,
                                float leaky_alpha)
{
    if (!blocks || layer_idx >= num_layers)
        return;

    blocks[layer_idx].type = CnnBlockType::Dense;
    blocks[layer_idx].dense.weights = weights;
    blocks[layer_idx].dense.bias = bias;
    blocks[layer_idx].dense.activation = activation;
    blocks[layer_idx].dense.leaky_alpha = leaky_alpha;
}

Tensor& CNNNetwork::forward(const Tensor& input, Arena& arena)
{
    static Tensor empty{};

    if (!IsValid() || num_layers == 0)
        return empty;

    Tensor current_input = input;

    for (uint32_t i = 0; i < num_layers; ++i)
    {
        switch (blocks[i].type)
        {
            case CnnBlockType::Conv2D:
            {
                const uint32_t out_h =
                    CalcOutputDim(current_input.shape[0], blocks[i].conv.conv.kernel_size, blocks[i].conv.conv.stride);
                const uint32_t out_w =
                    CalcOutputDim(current_input.shape[1], blocks[i].conv.conv.kernel_size, blocks[i].conv.conv.stride);
                const uint32_t out_c = static_cast<uint32_t>(blocks[i].conv.conv.out_channels);

                const std::array<uint32_t, 3> shape = {out_h, out_w, out_c};
                intermediate_outputs[i] = CreateND(arena, 3, shape);
                if (!intermediate_outputs[i].data)
                    return intermediate_outputs[i];

                blocks[i].conv.forward(current_input, intermediate_outputs[i]);
                current_input = intermediate_outputs[i];
                break;
            }
            case CnnBlockType::MaxPool2D:
            {
                const uint32_t out_h =
                    CalcOutputDim(current_input.shape[0], blocks[i].pool.pool_size, blocks[i].pool.stride);
                const uint32_t out_w =
                    CalcOutputDim(current_input.shape[1], blocks[i].pool.pool_size, blocks[i].pool.stride);
                const uint32_t out_c = current_input.shape[2];

                const std::array<uint32_t, 3> shape = {out_h, out_w, out_c};
                intermediate_outputs[i] = CreateND(arena, 3, shape);
                if (!intermediate_outputs[i].data)
                    return intermediate_outputs[i];

                blocks[i].pool.forward(current_input, intermediate_outputs[i]);
                current_input = intermediate_outputs[i];
                break;
            }
            case CnnBlockType::Flatten:
            {
                const uint32_t features = current_input.num_elements;
                intermediate_outputs[i] = Create2D(arena, 1, features);
                if (!intermediate_outputs[i].data)
                    return intermediate_outputs[i];

                FlattenNhwc(current_input, intermediate_outputs[i]);
                current_input = intermediate_outputs[i];
                break;
            }
            case CnnBlockType::Dense:
            {
                const uint32_t out_features = blocks[i].dense.weights.shape[1];
                intermediate_outputs[i] = Create2D(arena, 1, out_features);
                if (!intermediate_outputs[i].data)
                    return intermediate_outputs[i];

                blocks[i].dense.forward(current_input, intermediate_outputs[i]);
                current_input = intermediate_outputs[i];
                break;
            }
        }
    }

    return intermediate_outputs[num_layers - 1];
}
