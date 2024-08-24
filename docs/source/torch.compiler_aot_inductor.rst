AOTInductor: Ahead-Of-Time Compilation for Torch.Export-ed Models
=================================================================

.. warning::

    AOTInductor والميزات المتعلقة به هي في حالة النموذج الأولي وقد تخضع لتغييرات تكسير التوافق مع الإصدارات السابقة.

AOTInductor هو نسخة متخصصة من
`TorchInductor <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747>`__
، مصممة لمعالجة نماذج PyTorch المصدرة وتحسينها وإنتاج مكتبات مشتركة بالإضافة إلى
النتائج الأخرى ذات الصلة.
تم تصميم هذه القطع الأثرية المجمعة خصيصًا للنشر في بيئات غير Python،
والتي يتم استخدامها غالبًا لنشر الاستنتاجات على جانب الخادم.

في هذا البرنامج التعليمي، ستتعرف على عملية أخذ نموذج PyTorch، وتصديره،
وتجميعه في مكتبة مشتركة، وإجراء تنبؤات النموذج باستخدام C++.


تجميع النموذج
------------

باستخدام AOTInductor، يمكنك الاستمرار في إنشاء النموذج في Python. يوضح المثال التالي
كيفية استدعاء ``aot_compile`` لتحويل النموذج إلى مكتبة مشتركة.

تستخدم واجهة برمجة التطبيقات هذه ``torch.export`` لالتقاط النموذج في رسم بياني حسابي،
ثم تستخدم TorchInductor لتوليد .so الذي يمكن تشغيله في بيئة غير Python. للحصول على تفاصيل شاملة حول
واجهة برمجة تطبيقات ``torch._export.aot_compile``، يمكنك الرجوع إلى الكود
`هنا <https://github.com/pytorch/pytorch/blob/92cc52ab0e48a27d77becd37f1683fd442992120/torch/_export/__init__.py#L891-L900C9>`__.
للحصول على مزيد من التفاصيل حول ``torch.export``، يمكنك الرجوع إلى: المرجع: `وثائق torch.export <torch.export>`.

.. note::

   إذا كان لديك جهاز ممكّن لـ CUDA على جهازك وقمت بتثبيت PyTorch مع دعم CUDA،
   فسيقوم الكود التالي بتجميع النموذج إلى مكتبة مشتركة للتنفيذ باستخدام CUDA.
   وإلا، فسيتم تشغيل القطع الأثرية المجمعة على وحدة المعالجة المركزية. للحصول على أداء أفضل أثناء الاستدلال على وحدة المعالجة المركزية،
   يُقترح تمكين التجميد عن طريق تعيين ``export TORCHINDUCTOR_FREEZING=1``
   قبل تشغيل البرنامج النصي Python أدناه.

.. code-block:: python

    import os
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 16)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(16, 1)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x

    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Model().to(device=device)
        example_inputs=(torch.randn(8, 10, device=device),)
        batch_dim = torch.export.Dim("batch", min=1, max=1024)
        so_path = torch._export.aot_compile(
            model,
            example_inputs,
            # حدد البعد الأول لمدخل x كبعد ديناميكي
            dynamic_shapes={"x": {0: batch_dim}},
            # حدد مسار المكتبة المشتركة المولدة
            options={"aot_inductor.output_path": os.path.join(os.getcwd(), "model.so")},
        )

في هذا المثال التوضيحي، يتم استخدام معلمة ``Dim`` لتعيين البعد الأول لمتغير
المدخلات "x" كبعد ديناميكي. من الجدير بالذكر أن مسار المكتبة المجمّعة واسمها يظلان غير محددين،
مما يؤدي إلى تخزين المكتبة المشتركة في دليل مؤقت.
للوصول إلى هذا المسار من جانب C++، نقوم بحفظه في ملف لاستعادته لاحقًا ضمن كود C++.


الاستدلال في C++
---------------

بعد ذلك، نستخدم ملف C++ التالي ``inference.cpp`` لتحميل المكتبة المشتركة التي تم إنشاؤها في
الخطوة السابقة، مما يمكّننا من إجراء تنبؤات النموذج مباشرة ضمن بيئة C++.

.. note::

    يفترض مقتطف الكود التالي أن نظامك يحتوي على جهاز ممكّن لـ CUDA وأن نموذجك تم
تجميعه ليعمل على CUDA كما هو موضح سابقًا. في حالة عدم وجود وحدة معالجة رسومية (GPU)، من الضروري إجراء هذه التعديلات لتشغيلها على وحدة المعالجة المركزية:
1. تغيير ``model_container_runner_cuda.h`` إلى ``model_container_runner_cpu.h``
2. تغيير ``AOTIModelContainerRunnerCuda`` إلى ``AOTIModelContainerRunnerCpu``
3. تغيير ``at::kCUDA`` إلى ``at::kCPU``

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    #include <torch/torch.h>
    #include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>

    int main() {
        c10::InferenceMode mode;

        torch::inductor::AOTIModelContainerRunnerCuda runner("model.so");
        std::vector<torch::Tensor> inputs = {torch::randn({8, 10}, at::kCUDA)};
        std::vector<torch::Tensor> outputs = runner.run(inputs);
        std::cout << "Result from the first inference:"<< std::endl;
        std::cout << outputs[0] << std::endl;

        // يستخدم الاستدلال الثاني حجم دفعة مختلفًا ويعمل لأنه
        // تم تحديد ذلك البعد كبعد ديناميكي عند تجميع model.so.
        std::cout << "Result from the second inference:"<< std::endl;
        std::vector<torch::Tensor> inputs2 = {torch::randn({2, 10}, at::kCUDA)};
        std::cout << runner.run(inputs2)[0] << std::endl;

        return 0;
    }

للبناء ملف C++، يمكنك استخدام ملف ``CMakeLists.txt`` المرفق، والذي
يؤتمت عملية استدعاء ``python model.py`` لتجميع النموذج في الوقت المناسب وإنشاء ملف
``inference.cpp`` إلى ملف ثنائي قابل للتنفيذ باسم ``aoti_example``.

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
    project(aoti_example)

    find_package(Torch REQUIRED)

    add_executable(aoti_example inference.cpp model.so)

    add_custom_command(
        OUTPUT model.so
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/model.py
        DEPENDS model.py
    )

    target_link_libraries(aoti_example "${TORCH_LIBRARIES}")
    set_property(TARGET aoti_example PROPERTY CXX_STANDARD 17)


شريطة أن يشبه هيكل الدليل ما يلي، يمكنك تنفيذ الأوامر اللاحقة
لإنشاء الملف الثنائي. من المهم ملاحظة أن متغير ``CMAKE_PREFIX_PATH``
ضروري لـ CMake للعثور على مكتبة LibTorch، ويجب تعيينه إلى مسار مطلق.
يرجى ملاحظة أن مسارك قد يختلف عن المسار الموضح في هذا المثال.

.. code-block:: shell

    aoti_example/
        CMakeLists.txt
        inference.cpp
        model.py


.. code-block:: shell

    $ mkdir build
    $ cd build
    $ CMAKE_PREFIX_PATH=/path/to/python/install/site-packages/torch/share/cmake cmake ..
    $ cmake --build . --config Release

بعد إنشاء الملف الثنائي ``aoti_example`` في دليل ``build``، سيؤدي تنفيذه إلى
عرض نتائج مشابهة لما يلي:

.. code-block:: shell

    $ ./aoti_example
    Result from the first inference:
    0.4866
    0.5184
    0.4462
    0.4611
    0.4744
    0.4811
    0.4938
    0.4193
    [ CUDAFloatType{8,1} ]
    Result from the second inference:
    0.4883
    0.4703
    [ CUDAFloatType{2,1} ]