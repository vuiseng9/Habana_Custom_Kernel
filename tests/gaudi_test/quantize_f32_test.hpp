/**********************************************************************
Copyright (c) 2021 Habana Labs. All rights reserved.

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

#ifndef QUANTIZE_F32_TEST_HPP
#define QUANTIZE_F32_TEST_HPP

#include "test_base.hpp"
#include "tensor.h"
#include "quantize_f32_test.hpp"
#include "entry_points.hpp"
#include <algorithm>
#include "quantize_f32.hpp"

class QuantizeF32Test : public TestBase
{
public:
    QuantizeF32Test() {}
    ~QuantizeF32Test() {}
    int runTest(Gaudi_Kernel_Name_e NameofKernel);

    static void quantize_f32_reference_implementation(
        const float_5DTensor &input,
        const float_5DTensor &input_low,
        const float_5DTensor &input_range,
        const int levels,
        float_5DTensor &output, Gaudi_Kernel_Name_e mode);

private:
    QuantizeF32Test(const QuantizeF32Test& other) = delete;
    QuantizeF32Test& operator=(const QuantizeF32Test& other) = delete;

};


#endif /* QUANTIZE_F32_TEST_HPP */

