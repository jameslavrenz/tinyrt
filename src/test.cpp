#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include "arena.hpp"
#include "tensor.hpp"
#include "tensor_factory.hpp"
#include "mlp.hpp"
#include "cnn.hpp"
#include "model_loader.hpp"

using namespace TensorFactory;

namespace
{
    bool FileReadable(const char* path)
    {
        std::FILE* file = std::fopen(path, "rb");
        if (!file)
            return false;
        std::fclose(file);
        return true;
    }

    const char* ResolveModelPath(const char* rel_path, char* buffer, std::size_t buffer_size)
    {
        if (FileReadable(rel_path))
            return rel_path;

        std::snprintf(buffer, buffer_size, "../%s", rel_path);
        if (FileReadable(buffer))
            return buffer;

        return rel_path;
    }

    bool FloatEq(float a, float b, float eps = 1e-5f)
    {
        return std::fabs(a - b) <= eps;
    }

    bool CheckFloats(const float* actual, std::initializer_list<float> expected, const char* label)
    {
        uint32_t i = 0;
        for (float e : expected)
        {
            if (!FloatEq(actual[i], e))
            {
                std::cout << "FAIL " << label << ": expected " << e << " got " << actual[i] << " at index " << i << "\n";
                return false;
            }
            ++i;
        }
        std::cout << "PASS " << label << "\n";
        return true;
    }

    Tensor MakeNhwcInput(float* data, uint32_t h, uint32_t w, uint32_t c)
    {
        Tensor input{};
        input.data = data;
        input.type = DataType::Float32;
        input.rank = 3;
        input.shape[0] = h;
        input.shape[1] = w;
        input.shape[2] = c;
        input.stride[0] = w * c;
        input.stride[1] = c;
        input.stride[2] = 1;
        input.num_elements = h * w * c;
        input.bytes = input.num_elements * sizeof(float);
        return input;
    }

    bool LoadMlpFromFile(const char* json_path, Arena& arena, MLPNetwork*& network,
                         std::array<uint32_t, kMaxTensorRank>& input_shape, uint32_t& input_rank)
    {
        const ModelLoader::LoadResult result =
            ModelLoader::LoadMLP(json_path, arena, network, input_shape, input_rank);

        if (result.status != ModelLoader::LoadStatus::Ok || !network)
        {
            std::cout << "  Load failed (" << json_path << "): "
                      << (result.message ? result.message : "unknown error") << "\n";
            std::cout << "  Hint: run from the project root (where models/ lives).\n";
            return false;
        }
        return true;
    }

    bool LoadCnnFromFile(const char* json_path, Arena& arena, CNNNetwork*& network,
                         std::array<uint32_t, kMaxTensorRank>& input_shape, uint32_t& input_rank)
    {
        const ModelLoader::LoadResult result =
            ModelLoader::LoadCNN(json_path, arena, network, input_shape, input_rank);

        if (result.status != ModelLoader::LoadStatus::Ok || !network)
        {
            std::cout << "  Load failed (" << json_path << "): "
                      << (result.message ? result.message : "unknown error") << "\n";
            std::cout << "  Hint: run from the project root (where models/ lives).\n";
            return false;
        }
        return true;
    }

    bool RunMlpCase(const char* json_path,
                    std::initializer_list<float> input_values,
                    std::initializer_list<float> expected,
                    const char* label)
    {
        char path_buffer[ModelLoader::kMaxPathLen] = {};
        const char* resolved = ResolveModelPath(json_path, path_buffer, sizeof(path_buffer));

        unsigned char buffer[8192];
        Arena arena;
        arena.init(buffer, 8192);

        MLPNetwork* mlp = nullptr;
        std::array<uint32_t, kMaxTensorRank> input_shape{};
        uint32_t input_rank = 0;

        if (!LoadMlpFromFile(resolved, arena, mlp, input_shape, input_rank))
            return false;

        Tensor input = Create2D(arena, input_shape[0], input_shape[1]);
        Fill(input, input_values);

        const uint32_t output_cols = mlp->GetLayer(1).weights.shape[1];
        Tensor output = Create2D(arena, input_shape[0], output_cols);

        std::cout << "Model: " << resolved << "\n";
        PrintLabeled("Input", input);

        mlp->forward(input, output, arena);

        PrintLabeled("Output", output);

        return CheckFloats(static_cast<const float*>(output.data), expected, label);
    }

    bool RunCnnCase(const char* json_path,
                    float* input_data,
                    std::initializer_list<float> expected,
                    const char* label)
    {
        char path_buffer[ModelLoader::kMaxPathLen] = {};
        const char* resolved = ResolveModelPath(json_path, path_buffer, sizeof(path_buffer));

        unsigned char buffer[8192];
        Arena arena;
        arena.init(buffer, 8192);

        CNNNetwork* cnn = nullptr;
        std::array<uint32_t, kMaxTensorRank> input_shape{};
        uint32_t input_rank = 0;

        if (!LoadCnnFromFile(resolved, arena, cnn, input_shape, input_rank))
            return false;

        Tensor input = MakeNhwcInput(input_data, input_shape[0], input_shape[1], input_shape[2]);

        std::cout << "Model: " << resolved << "\n";
        PrintLabeled("Input", input);

        Tensor& output = cnn->forward(input, arena);

        PrintLabeled("Output", output);

        return CheckFloats(static_cast<const float*>(output.data), expected, label);
    }
}

void test_mlp()
{
    std::cout << "\n============================\n";
    std::cout << " MLP TEST\n";
    std::cout << "============================\n";

    RunMlpCase("models/test_mlp.json",
               {1.0f, 2.0f},
               {3.0f, 3.0f},
               "MLP 2-layer output");
}

void test_cnn()
{
    std::cout << "\n============================\n";
    std::cout << " CNN TESTS\n";
    std::cout << "============================\n";

    float input_2x2x2[] = {
        1.0f, 10.0f,
        2.0f, 20.0f,
        3.0f, 30.0f,
        4.0f, 40.0f
    };

    RunCnnCase("models/test_cnn.json",
               input_2x2x2,
               {21.0f, 42.0f, 42.0f, 84.0f, 63.0f, 126.0f, 84.0f, 168.0f},
               "CNN 2-layer (channel stacking)");

    float input_4x4x1[16] = {
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1
    };

    RunCnnCase("models/cnn_4x4_single.json",
               input_4x4x1,
               {4.0f, 4.0f, 4.0f, 4.0f},
               "CNN 3x3 single-layer (spatial conv)");
}
