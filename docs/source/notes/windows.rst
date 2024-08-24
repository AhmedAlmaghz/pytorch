أسئلة شائعة حول Windows
====================

بناء من المصدر
-------------

تضمين المكونات الاختيارية
^^^^^^^^^^^^^^^^^^^^

هناك مكونان مدعومان لنظام Windows PyTorch: MKL و MAGMA. فيما يلي خطوات البناء باستخدامها.

.. code-block:: bat

    REM تأكد من تثبيت 7z و curl.

    REM تنزيل ملفات MKL
    curl https://s3.amazonaws.com/ossci-windows/mkl_2020.2.254.7z -k -O
    7z x -aoa mkl_2020.2.254.7z -omkl

    REM تنزيل ملفات MAGMA
    REM الإصدارات المتاحة:
    REM 2.5.4 (CUDA 10.1 10.2 11.0 11.1) x (Debug Release)
    REM 2.5.3 (CUDA 10.1 10.2 11.0) x (Debug Release)
    REM 2.5.2 (CUDA 9.2 10.0 10.1 10.2) x (Debug Release)
    REM 2.5.1 (CUDA 9.2 10.0 10.1 10.2) x (Debug Release)
    set CUDA_PREFIX=cuda102
    set CONFIG=release
    curl -k https://s3.amazonaws.com/ossci-windows/magma_2.5.4_%CUDA_PREFIX%_%CONFIG%.7z -o magma.7z
    7z x -aoa magma.7z -omagma

    REM تعيين متغيرات البيئة الأساسية
    set "CMAKE_INCLUDE_PATH=%cd%\mkl\include"
    set "LIB=%cd%\mkl\lib;%LIB%"
    set "MAGMA_HOME=%cd%\magma"

تسريع بناء CUDA لنظام Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

لا يدعم Visual Studio حاليًا المهام المخصصة الموازية.
وكبديل، يمكننا استخدام ``Ninja`` لموازاة
مهام بناء CUDA. يمكن استخدامه عن طريق كتابة بضع سطور من التعليمات البرمجية فقط.

.. code-block:: bat

    REM دعونا نقوم بتثبيت ninja أولاً.
    pip install ninja

    REM تعيينه كمولد cmake
    set CMAKE_GENERATOR=Ninja

سكريبت التثبيت بنقرة واحدة
^^^^^^^^^^^^^^^^^^^^^^

يمكنك إلقاء نظرة على `هذه المجموعة من السكريبتات
<https://github.com/peterjc123/pytorch-scripts>`_.
سيقودك إلى الطريق.

امتداد
---------

امتداد CFFI
^^^^^^^^^^^^^^

إن الدعم لامتداد CFFI تجريبي للغاية. يجب عليك تحديد
مكتبات ``إضافية`` في كائن ``Extension`` لجعله يبني على
Windows.

.. code-block:: python

   ffi = create_extension(
       '_ext.my_lib',
       headers=headers,
       sources=sources,
       define_macros=defines,
       relative_to=__file__,
       with_cuda=with_cuda,
       extra_compile_args=["-std=c99"],
       libraries=['ATen', '_C'] # أضف مكتبات cuda عند الضرورة، مثل cudart
   )

امتداد Cpp
^^^^^^^^^^^^^

هذا النوع من الامتداد له دعم أفضل مقارنة
بالسابق. ومع ذلك، لا يزال بحاجة إلى بعض التكوين اليدوي. أولاً، يجب عليك فتح
**x86_x64 أدوات متقاطعة سطر الأوامر لـ VS 2017**.
وبعد ذلك، يمكنك بدء عملية التجميع الخاصة بك.

التثبيت
-------

تعذر العثور على الحزمة في قناة win-32.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bat

    فشل حل البيئة:

    PackagesNotFoundError: لا تتوفر الحزم التالية من القنوات الحالية:

    - pytorch

    القنوات الحالية:
    - https://conda.anaconda.org/pytorch/win-32
    - https://conda.anaconda.org/pytorch/noarch
    - https://repo.continuum.io/pkgs/main/win-32
    - https://repo.continuum.io/pkgs/main/noarch
    - https://repo.continuum.io/pkgs/free/win-32
    - https://repo.continuum.io/pkgs/free/noarch
    - https://repo.continuum.io/pkgs/r/win-32
    - https://repo.continuum.io/pkgs/r/noarch
    - https://repo.continuum.io/pkgs/pro/win-32
    - https://repo.continuum.io/pkgs/pro/noarch
    - https://repo.continuum.io/pkgs/msys2/win-32
    - https://repo.continuum.io/pkgs/msys2/noarch

لا يعمل PyTorch على نظام 32 بت. يرجى استخدام إصدار Windows و
Python 64 بت.


خطأ الاستيراد
^^^^^^^^^^^^

.. code-block:: python

    from torch._C import *

    ImportError: فشل تحميل DLL: لم يتم العثور على الوحدة النمطية المحددة.


تسبب المشكلة في فقدان الملفات الأساسية. في الواقع،
نحن ندرج جميع الملفات الأساسية تقريبًا التي تحتاجها PyTorch لحزمة conda
باستثناء VC2017 القابلة لإعادة التوزيع وبعض مكتبات MKL.
يمكنك حل هذه المشكلة عن طريق كتابة الأمر التالي.

.. code-block:: bat

    conda install -c peterjc123 vc vs2017_runtime
    conda install mkl_fft intel_openmp numpy mkl

بالنسبة لحزمة العجلات، نظرًا لأننا لم نقم بتعبئة بعض المكتبات وملفات VC2017
القابلة لإعادة التوزيع، يرجى التأكد من تثبيتها يدويًا.
يمكن تنزيل `مثبت VC 2017 القابل لإعادة التوزيع
<https://aka.ms/vs/15/release/VC_redist.x64.exe>`_.
وعليك أيضًا الانتباه إلى تثبيت Numpy الخاص بك. تأكد من أنه
يستخدم MKL بدلاً من OpenBLAS. قد تكتب الأمر التالي.

.. code-block:: bat

    pip install numpy mkl intel-openmp mkl_fft

قد يكون السبب المحتمل الآخر هو استخدامك للإصدار GPU دون بطاقات رسومات NVIDIA.
يرجى استبدال حزمة GPU الخاصة بك بالإصدار CPU.

.. code-block:: python

    from torch._C import *

    ImportError: فشل تحميل DLL: لا يمكن تشغيل نظام التشغيل %1.


هذه في الواقع مشكلة في Anaconda. عندما تقوم بتهيئة بيئتك باستخدام قناة conda-forge،
ستظهر هذه المشكلة. يمكنك إصلاح
مكتبات intel-openmp من خلال هذا الأمر.

.. code-block:: bat

    conda install -c defaults intel-openmp -f


الاستخدام (تعدد العمليات)
----------------------

خطأ تعدد العمليات دون حماية شرط if
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    RuntimeError:
           تم إجراء محاولة لبدء عملية جديدة قبل أن تنتهي العملية الحالية من مرحلة التمهيد الخاصة بها.

       هذا يعني على الأرجح أنك لا تستخدم fork لبدء عمليات الطفل الخاصة بك وأنك نسيت استخدام الاصطلاح الصحيح
       في الوحدة النمطية الرئيسية:

           if __name__ == '__main__':
               freeze_support()
               ...

       يمكن حذف سطر "freeze_support()" إذا لم يكن البرنامج
       سيتم تجميده لإنتاج ملف تنفيذي.

يختلف تنفيذ ``multiprocessing`` على نظام Windows، والذي
يستخدم ``spawn`` بدلاً من ``fork``. لذلك يجب علينا لف التعليمات البرمجية باستخدام
شرط if لحماية التعليمات البرمجية من التنفيذ عدة مرات. قم بإعادة هيكلة التعليمات البرمجية الخاصة بك إلى
هيكل التالي.

.. code-block:: python

    import torch

    def main()
        for i, data in enumerate(dataloader):
            # قم بشيء ما هنا

    if __name__ == '__main__':
        main()


خطأ تعدد العمليات "كسر الأنبوب"
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    ForkingPickler(file, protocol).dump(obj)

    BrokenPipeError: [Errno 32] كسر الأنبوب

تحدث هذه المشكلة عندما تنتهي عملية الطفل قبل أن تنتهي عملية الوالد
من إرسال البيانات. قد يكون هناك خطأ ما في التعليمات البرمجية الخاصة بك. يمكنك
تصحيح التعليمات البرمجية الخاصة بك عن طريق تقليل ``num_worker`` من
:class:`~torch.utils.data.DataLoader` إلى الصفر ومعرفة ما إذا كانت المشكلة مستمرة.

خطأ تعدد العمليات "إيقاف التشغيل"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    لم يتمكن من فتح ملف الخريطة المشتركة: <torch_14808_1591070686>، رمز الخطأ: <1455> في torch\lib\TH\THAllocator.c:154

    [windows] توقف التشغيل

يرجى تحديث برنامج تشغيل الرسومات الخاص بك. إذا استمرت هذه المشكلة، فقد يكون ذلك بسبب أن
بطاقة الرسومات لديك قديمة جدًا أو أن الحساب ثقيل جدًا لبطاقتك. يرجى
تحديث إعدادات TDR وفقًا لهذا `المنشور
<https://www.pugetsystems.com/labs/hpc/Working-around-TDR-in-Windows-for-a-better-GPU-computing-experience-777/>`_.

عمليات IPC CUDA
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   فشل THCudaCheck ملف=torch\csrc\generic\StorageSharing.cpp line=252 error=63: فشل استدعاء نظام التشغيل أو عدم دعم العملية على نظام التشغيل هذا

إنها غير مدعومة على Windows. لا يمكن لشيء مثل القيام بتعدد العمليات على CUDA
tensors أن ينجح، هناك بديلان لهذا.

1. لا تستخدم ``multiprocessing``. قم بتعيين ``num_worker`` من
:class:`~torch.utils.data.DataLoader` إلى الصفر.

2. شارك tensors CPU بدلاً من ذلك. تأكد من أن مجموعة البيانات المخصصة الخاصة بك
:class:`~torch.utils.data.DataSet` تعيد tensors CPU.