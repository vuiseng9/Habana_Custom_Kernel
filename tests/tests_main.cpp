/**********************************************************************
Copyright (c) 2022 Habana Labs.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include <iostream>
#include "filter_fwd_2d_bf16_test.hpp"
#include "softmax_bf16_test.hpp"
#include "cast_gaudi_test.hpp"
#include "batchnorm_f32_test.hpp"
#include "leakyrelu_f32_gaudi_test.hpp"
#include "sparse_lengths_sum_bf16_test.hpp"
#include "customdiv_fwd_f32_test.hpp"
#include "relu6_all_test.hpp"
#include "matrix_mul_fwd_f32_test.hpp"
#include "spatial_conv_f32_test.hpp"
#include "sin_f32_test.hpp"
#include "add_f32_test.hpp"
#include "avg_pool_2d_f32_test.hpp"
#include "avg_pool_2d_f32_gaudi2_test.hpp"
#include "cast_f16_to_i16_gaudi2_test.hpp"
#include "quantize_f32_test.hpp"

int main(int argc, char** argv)
{
    int result = 0;
    static int testCount = 0;

    if(argc == 2 && ((strcmp(argv[1], "--help") ==0) || (strcmp(argv[1],"-h") ==0)))
    {
        std::cout << argv[0] << " " << "[options]" << std::endl <<
            "Options:" << std::endl <<
            "N/A                        Run all test cases" << std::endl <<
            "-h | --help                Print this help" << std::endl <<
            "-d | --device <DeviceName> Run only kernels for the DeviceName" << std::endl <<
            "-t | --test  <TestName>    Run <TestName>> only   " << std::endl <<
            "DeviceName:" << std::endl <<
            "Gaudi                      Run all Gaudi kernels only   " << std::endl <<
            "TestName:" << std::endl <<
            "FilterFwd2DBF16Test        Run FilterFwd2DBF16Test only   " << std::endl <<
            "SoftMaxBF16Test            Run SoftMaxBF16Test only   " << std::endl <<
            "CastGaudiTest              Run CastGaudiTest only   " << std::endl <<
            "BatchNormF32Test           Run BatchNormF32Test only   " << std::endl <<
            "LeakyReluF32GaudiTest      Run LeakyReluF32GaudiTest only   " << std::endl <<
            "SparseLengthsBF16Test      Run SparseLengthsBF16Test only   " << std::endl <<
            "CustomdivFwdF32Test        Run CustomdivFwdF32Test only   " << std::endl <<
            "Relu6FwdF32                Run Relu6FwdF32 only   " << std::endl <<
            "Relu6BwdF32                Run Relu6BwdF32 only   " << std::endl <<
            "Relu6FwdBF16               Run Relu6FwdBF16 only   " << std::endl <<
            "Relu6BwdBF16               Run Relu6BwdBF16 only   " << std::endl <<
            "ReluFwdF32                 Run ReluFwdF32 only   " << std::endl <<
            "ReluBwdF32                 Run ReluBwdF32 only   " << std::endl <<
            "ReluFwdBF16                Run ReluFwdBF16 only   " << std::endl <<
            "ReluBwdBF16                Run ReluBwdBF16 only   " << std::endl <<
            "MatrixMulFwdF32Test        Run MatrixMulFwdF32Test only   " << std::endl <<
            "SpatialConvF32Test         Run SpatialConvF32Test only   " << std::endl <<
            "SinF32Test                 Run SinF32Test only   " << std::endl <<
            "AddF32Test                 Run AddF32Test only   " << std::endl <<
            "AvgPool2DFwdF32Test        Run AvgPool2DFwdF32Test only   " << std::endl <<
            "AvgPool2DBwdF32Test        Run AvgPool2DBwdF32Test only   " << std::endl <<

            "AvgPool2DFwdF32Gaudi2Test  Run AvgPool2DFwdF32Gaudi2Test only   " << std::endl <<
            "AvgPool2DBwdF32Gaudi2Test  Run AvgPool2DBwdF32Gaudi2Test only   " << std::endl <<
            "CastF16toI16Gaudi2Test     Run CastF16toI16Gaudi2Test only   " << std::endl <<
            "QuantizeFwdF32             Run QuantizeFwdF32 only   " << std::endl <<
            "QuantizeBwdF32             Run QuantizeBwdF32 only   " << std::endl;
        exit(0);
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"FilterFwd2DBF16Test") ==0))))
    {
        FilterFwd2DBF16Test test_bf16;
        test_bf16.SetUp();
        result = test_bf16.runTest();
        test_bf16.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"SoftMaxBF16Test") ==0))))
    {
        SoftMaxBF16Test testSoftMaxBF16;
        testSoftMaxBF16.SetUp();
        result = testSoftMaxBF16.runTest();
        testSoftMaxBF16.TearDown();
        testCount += 2;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"CastGaudiTest") ==0))))
    {
        CastGaudiTest testCaseGaudi;
        testCaseGaudi.SetUp();
        result = testCaseGaudi.runTest();
        testCaseGaudi.TearDown();
        testCount += 2;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"BatchNormF32Test") ==0))))
    {
        BatchNormF32Test testBatchNorm;
        testBatchNorm.SetUp();
        result = testBatchNorm.runTest();
        testBatchNorm.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"LeakyReluF32GaudiTest") ==0))))
    {
        LeakyReluF32GaudiTest testLeakyReluGaudi;
        testLeakyReluGaudi.SetUp();
        result = testLeakyReluGaudi.runTest();
        testLeakyReluGaudi.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"SparseLengthsBF16Test") ==0))))
    {
        SparseLengthsSumBF16Test testSparseLenGaudi;
        testSparseLenGaudi.SetUp();
        result = testSparseLenGaudi.runTest();
        testSparseLenGaudi.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"CustomdivFwdF32Test") ==0))))
    {
        CustomdivFwdF32Test testCustomDivFwdF32;
        testCustomDivFwdF32.SetUp();
        result = testCustomDivFwdF32.runTest();
        testCustomDivFwdF32.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    Relu6AllTest testRelu6;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"Relu6FwdF32") ==0))))
    {
        testRelu6.SetUp();
        result = testRelu6.runTest(GAUDI_KERNEL_RELU6_FWD_F32);
        testRelu6.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"Relu6BwdF32") ==0))))
    {
        testRelu6.SetUp();
        result = testRelu6.runTest(GAUDI_KERNEL_RELU6_BWD_F32);
        testRelu6.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"Relu6FwdBF16") ==0))))
    {
        testRelu6.SetUp();
        result = testRelu6.runTest(GAUDI_KERNEL_RELU6_FWD_BF16);
        testRelu6.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"Relu6BwdBF16") ==0))))
    {
        testRelu6.SetUp();
        result = testRelu6.runTest(GAUDI_KERNEL_RELU6_BWD_BF16);
        testRelu6.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    Relu6AllTest testRelu;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"ReluFwdF32") ==0))))
    {
        testRelu.SetUp();
        result = testRelu.runTest(GAUDI_KERNEL_RELU_FWD_F32);
        testRelu.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"ReluBwdF32") ==0))))
    {
        testRelu.SetUp();
        result = testRelu.runTest(GAUDI_KERNEL_RELU_BWD_F32);
        testRelu.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"ReluFwdBF16") ==0))))
    {
        testRelu.SetUp();
        result = testRelu.runTest(GAUDI_KERNEL_RELU_FWD_BF16);
        testRelu.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"ReluBwdBF16") ==0))))
    {
        testRelu.SetUp();
        result = testRelu.runTest(GAUDI_KERNEL_RELU_BWD_BF16);
        testRelu.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    QuantizeF32Test testQuantize;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"QuantizeFwdF32") ==0))))
    {
        testQuantize.SetUp();
        result = testQuantize.runTest(GAUDI_KERNEL_QUANTIZE_FWD_F32);
        testQuantize.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"QuantizeBwdF32") ==0))))
    {
        testQuantize.SetUp();
        result = testQuantize.runTest(GAUDI_KERNEL_QUANTIZE_BWD_F32);
        testQuantize.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"MatrixMulFwdF32Test") ==0))))
    {
        MatrixMulFwdF32Test testMatrixMulFwdF32;
        testMatrixMulFwdF32.SetUp();
        result = testMatrixMulFwdF32.runTest();
        testMatrixMulFwdF32.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"SpatialConvF32Test") ==0))))
    {
        SpatialConvF32Test spatialConv;
        spatialConv.SetUp();
        result = spatialConv.runTest();
        spatialConv.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"SinF32Test") ==0))))
    {
        SinF32Test sinf32ins;
        sinf32ins.SetUp();
        result = sinf32ins.runTest();
        sinf32ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"AddF32Test") ==0))))
    {
        AddF32Test addf32ins;
        addf32ins.SetUp();
        result = addf32ins.runTest();
        addf32ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    AvgPool2DF32Test avgpool2df32ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"AvgPool2DFwdF32Test") ==0))))
    {
        avgpool2df32ins.SetUp();
        result = avgpool2df32ins.runTest(GAUDI_KERNEL_AVG_POOL_2D_FWD_F32);
        avgpool2df32ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"AvgPool2DBwdF32Test") ==0))))
    {
        avgpool2df32ins.SetUp();
        result = avgpool2df32ins.runTest(GAUDI_KERNEL_AVG_POOL_2D_BWD_F32);
        avgpool2df32ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    AvgPool2DF32Gaudi2Test avgpool2df32Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"AvgPool2DFwdF32Gaudi2Test") ==0))))
    {
        avgpool2df32Gaudi2ins.SetUp();
        result = avgpool2df32Gaudi2ins.runTest(GAUDI2_KERNEL_AVG_POOL_2D_FWD_F32);
        avgpool2df32Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"AvgPool2DBwdF32Gaudi2Test") ==0))))
    {
        avgpool2df32Gaudi2ins.SetUp();
        result = avgpool2df32Gaudi2ins.runTest(GAUDI2_KERNEL_AVG_POOL_2D_BWD_F32);
        avgpool2df32Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    CastF16toI16Gaudi2Test castf16tpi16Gaudi2ins;
    if(argc == 1 ||
        (argc == 3 && (((strcmp(argv[1], "--device") ==0) || (strcmp(argv[1], "-d") ==0))
        && (strcmp(argv[2],"Gaudi2") ==0)))  ||
        (argc == 3 && (((strcmp(argv[1], "--test") ==0) || (strcmp(argv[1], "-t") ==0))
        && (strcmp(argv[2],"CastF16toI16Gaudi2Test") ==0))))
    {
        castf16tpi16Gaudi2ins.SetUp();
        result = castf16tpi16Gaudi2ins.runTest();
        castf16tpi16Gaudi2ins.TearDown();
        testCount ++;
        if (result != 0)
        {
            return result;
        }
    }

    std::cout << "All " << testCount  <<" tests pass!" <<std::endl;
    return 0;
}
