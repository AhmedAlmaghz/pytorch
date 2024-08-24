TorchScript-based ONNX Exporter
------------------------------------

يقوم المصدر القائم على TorchScript بتصدير النماذج إلى تنسيق ONNX.

.. currentmodule:: torch.onnx

.. function:: export(model, args, f, export_params=True, verbose=False, training=False, input_names=None, output_names=None, operator_export_type=None, opset_version=None, do_constant_folding=True, example_outputs=None, dynamic_axes=None, custom_opsets=None)

   تصدير النموذج إلى ملف ONNX.

   الحجة:
      النموذج: النموذج الذي سيتم تصديره.
      args: قائمة من باقي المدخلات المطلوبة من قبل النموذج.
      f: اسم ملف الإخراج.
      export_params=True: إذا كان يجب تصدير معلمات النموذج.
      verbose=False: إذا كان يجب طباعة الرسائل أثناء التصدير.
      training=False: إذا كان النموذج في الوضع التدريبي.
      input_names=None: قائمة من أسماء المدخلات. إذا لم يتم توفيرها، فسيتم استخدام أسماء مثل "input_0".
      output_names=None: قائمة من أسماء الإخراج. إذا لم يتم توفيرها، فسيتم استخدام أسماء مثل "output_0".
      operator_export_type=None: نوع التصدير. يمكن أن يكون "ONNX" أو "ONNX_ATEN".
      opset_version=None: إصدار مجموعة العمليات. إذا لم يتم توفيره، سيتم استخدام الإصدار الافتراضي.
      do_constant_folding=True: إذا كان يجب تطبيق الثنيات الثابتة.
      example_outputs=None: إخراج نموذجي يستخدم لتحديد الأبعاد الديناميكية.
      dynamic_axes=None: قائمة من المحاور الديناميكية لكل مدخل وإخراج.
      custom_opsets=None: مجموعة من مجموعات العمليات المخصصة.

.. function:: _export(model, args, f, export_params=True, verbose=False, training=False, input_names=None, output_names=None, operator_export_type=None, opset_version=None, do_constant_folding=True, example_outputs=None, dynamic_axes=None, custom_opsets=None)

   نفس التصدير ولكن لا يتحقق من وجود ملف الإخراج.

.. function:: register_custom_op_symbolic(name, symbolic_function, symbolic_function_backward=None)

   تسجيل دالة رمزية لتصدير المشغل المخصص.

   الحجة:
      name: اسم المشغل المخصص.
      symbolic_function: الدالة الرمزية التي تحدد سلوك المشغل المخصص.
      symbolic_function_backward: الدالة الرمزية للتفاضل التلقائي للمشغل المخصص. اختياري.

.. function:: register_opset_version(name, version)

   تسجيل إصدار مجموعة العمليات لمجموعة عمليات ONNX.

   الحجة:
      name: اسم مجموعة العمليات.
      version: إصدار مجموعة العمليات.

.. function:: set_operator_export_type(operator_export_type)

   تعيين نوع التصدير.

   الحجة:
      operator_export_type: نوع التصدير. يمكن أن يكون "ONNX" أو "ONNX_ATEN".

.. function:: set_training(training)

   تعيين وضع التدريب للنموذج.

   الحجة:
      training: إذا كان النموذج في الوضع التدريبي.

.. function:: is_in_onnx_export()

   التحقق مما إذا كنا في عملية تصدير ONNX.

.. function:: handle_custom_op(ctx, name, inputs)

   التعامل مع المشغل المخصص أثناء التصدير.

   الحجة:
      ctx: سياق التصدير.
      name: اسم المشغل المخصص.
      inputs: قائمة من المدخلات للمشغل المخصص.

.. function:: register_standard_custom_ops()

   تسجيل المشغلين المخصصين القياسيين.

.. function:: register_custom_ops(custom_op_dict)

   تسجيل المشغلين المخصصين من القاموس.

   الحجة:
      custom_op_dict: قاموس من أسماء المشغلين المخصصين إلى الدوال الرمزية.

.. function:: register_custom_op(name, symbolic_function, symbolic_function_backward=None)

   تسجيل مشغل مخصص. نفس register_custom_op_symbolic.

.. function:: prepare_torch_op(ctx, torch_op, args, kwargs)

   إعداد مشغل PyTorch للتصدير.

   الحجة:
      ctx: سياق التصدير.
      torch_op: مشغل PyTorch.
      args: قائمة من المدخلات للمشغل.
      kwargs: قاموس من الكلمات الرئيسية للمشغل.

.. function:: prepare_aten_op(ctx, aten_op, args, kwargs)

   إعداد مشغل ATen للتصدير.

   الحجة:
      ctx: سياق التصدية.
      aten_op: مشغل ATen.
      args: قائمة من المدخلات للمشغل.
      kwargs: قاموس من الكلمات الرئيسية للمشغل.

.. function:: export_to_pretty_string(model, args, operator_export_type=None, opset_version=None, do_constant_folding=True, dynamic_axes=None, custom_opsets=None)

   تصدير النموذج إلى سلسلة ONNX جميلة.

   الحجة:
      model: النموذج الذي سيتم تصديره.
      args: قائمة من باقي المدخلات المطلوبة من قبل النموذج.
      operator_export_type=None: نوع التصدير. يمكن أن يكون "ONNX" أو "ONNX_ATEN".
      opset_version=None: إصدار مجموعة العمليات. إذا لم يتم توفيره، سيتم استخدام الإصدار الافتراضي.
      do_constant_folding=True: إذا كان يجب تطبيق الثنيات الثابتة.
      dynamic_axes=None: قائمة من المحاور الديناميكية لكل مدخل وإخراج.
      custom_opsets=None: مجموعة من مجموعات العمليات المخصصة.

.. function:: use_stim_exporter()

   استخدام مصدر STIM.

.. function:: use_torch_exporter()

   استخدام مصدر PyTorch.

.. currentmodule:: torch.onnx.symbolic_helper

.. function:: parse_args(inputs, kwargs)

   تحليل المدخلات والوسيطات الإضافية.

   الحجة:
      inputs: قائمة من المدخلات.
      kwargs: قاموس من الكلمات الرئيسية.

.. function:: _parse_arg(arg)

   تحليل وسيطة واحدة.

   الحجة:
      arg: وسيطة واحدة.

.. function:: _parse_args(inputs, kwargs)

   تحليل المدخلات والوسيطات الإضافية.

.. currentmodule:: torch.onnx.symbolic

.. function:: call_module(module, inputs, kwargs)

   استدعاء الوحدة النمطية مع المدخلات والوسيطات الإضافية.

   الحجة:
      module: الوحدة النمطية التي سيتم استدعاؤها.
      inputs: قائمة من المدخلات.
      kwargs: قاموس من الكلمات الرئيسية.

.. function:: call_function(function, inputs, kwargs)

   استدعاء الدالة مع المدخلات والوسيطات الإضافية.

   الحجة:
      function: الدالة التي سيتم استدعاؤها.
      inputs: قائمة من المدخلات.
      kwargs: قاموس من الكلمات الرئيسية.

.. function:: call_script_function(function, inputs, kwargs)

   استدعاء دالة النص البرمجي مع المدخلات والوسيطات الإضافية.

   الحجة:
      function: دالة النص البرمجي التي سيتم استدعاؤها.
      inputs: قائمة من المدخلات.
      kwargs: قاموس من الكلمات الرئيسية.

.. function:: call_method(obj, method, inputs, kwargs)

   استدعاء طريقة الكائن مع المدخلات والوسيطات الإضافية.

   الحجة:
      obj: كائن به طريقة.
      method: طريقة لاستدعاء.
      inputs: قائمة من المدخلات.
      kwargs: قاموس من الكلمات الرئيسية.

.. function:: call_script_method(obj, method, inputs, kwargs)

   استدعاء طريقة نص برمجي للكائن مع المدخلات والوسيطات الإضافية.

   الحجة.
      obj: كائن به طريقة نص برمجي.
      method: طريقة نص برمجي لاستدعاء.
      inputs: قائمة من المدخلات.
      kwargs: قاموس من الكلمات الرئيسية.

.. function:: call_module_if_available(module, inputs, kwargs)

   استدعاء الوحدة النمطية إذا كانت متاحة مع المدخلات والوسيطات الإضافية.

   الحجة:
      module: الوحدة النمطية التي سيتم استدعاؤها إذا كانت متاحة.
      inputs: قائمة من المدخلات.
      kwargs: قاموس من الكلمات الرئيسية.

.. function:: call_function_if_available(function, inputs, kwargs)

   استدعاء الدالة إذا كانت متاحة مع المدخلات والوسيطات الإضافية.

   الحجة:
      function: الدالة التي سيتم استدعاؤها إذا كانت متاحة.
      inputs: قائمة من المدخلات.
      kwargs. قاموس من الكلمات الرئيسية.

.. function:: call_script_function_if_available(function, inputs, kwargs)

   استدعاء دالة النص البرمجي إذا كانت متاحة مع المدخلات والوسيطات الإضافية.

   الحجة:
      function: دالة النص البرمجي التي سيتم استدعاؤها إذا كانت متاحة.
      inputs: قائمة من المدخلات.
      kwargs: قاموس من الكلمات الرئيسية.

.. function:: call_method_if_available(obj, method, inputs, kwargs)

   استدعاء طريقة الكائن إذا كانت متاحة مع المدخلات والوسيطات الإضافية.

   الحجة:
      obj: كائن به طريقة.
      method: طريقة لاستدعاء إذا كانت متاحة.
      inputs: قائمة من المدخلات.
      kwargs: قاموس من الكلمات الرئيسية.

.. function:: call_script_method_if_available(obj, method, inputs, kwargs)

   استدعاء طريقة نص برمجي للكائن إذا كانت متاحة مع المدخلات والوسيطات الإضافية.

   الحجة:
      obj: كائن به طريقة نص برمجي.
      method: طريقة نص برمجي لاستدعاء إذا كانت متاحة.
      inputs inputs: قائمة من المدخلات.
      kwargs: قاموس من الكلمات الرئيسية.

.. currentmodule:: torch.onnx.symbolic_opset

.. function:: add(g, input1, input2, **kwargs)

   إضافة مدخلات.

.. function:: addmm(g, input1, mat1, mat2, **kwargs)

   إضافة مصفوفتين.

.. function:: arange(g, start, end, **kwargs)

   إنشاء مصفوفة مع قيم تتراوح من البداية إلى النهاية.

.. function:: argmax(g, input, dim, **kwargs)

   إرجاع الفهرس ذو القيمة القصوى على البعد.

.. function:: argmin(g, input, dim, **kwargs)

   إرجاع الفهرس ذو القيمة الدنيا على البعد.

.. function

تصدير ONNX القائم على TorchScript
------------------------------------

يقوم مصدر ONNX القائم على TorchScript بتصدير النماذج إلى تنسيق ONNX.

.. currentmodule:: torch.onnx

.. function:: export(model, args, f, export_params=True, verbose=False, training=False, input_names=None, output_names=None, operator_export_type=None, opset_version=瓦解, do_constant_folding=True, example_outputs=None, dynamic_axes=None, custom_opsets=None)

   تصدير النموذج إلى ملف ONNX.

   الحجة:
      النموذج: النموذج الذي سيتم تصديره.
      args: قائمة من باقي المدخلات المطلوبة من قبل النموذج.
      f: اسم ملف الإخراج.
      export_params=True: إذا كان يجب تصدير معلمات النموذج.
      verbose=False: إذا كان يجب طباعة الرسائل أثناء التصدير.
      training=False: إذا كان النموذج في الوضع التدريبي.
      input_names=None: قائمة من أسماء المدخلات. إذا لم يتم توفيرها، فسيتم استخدام أسماء مثل "input_0".
      output_names=None: قائمة من أسماء الإخراج. إذا لم يتم توفيرها، فسيتم استخدام أسماء مثل "output_0".
      operator_export_type=None: نوع التصدير، يمكن أن يكون "ONNX" أو "ONNX_ATEN".
      opset_version=None: إصدار مجموعة العمليات. إذا لم يتم توفيره، سيتم استخدام الإصدار الافتراضي.
      do_constant_folding=True: إذا كان يجب تطبيق الثنيات الثابتة.
      example_outputs=None: إخراج نموذجي يستخدم لتحديد الأبعاد الديناميكية.
      dynamic_axes=None: قائمة من المحاور الديناميكية لكل مدخل وإخراج.
      custom_opsets=None: مجموعة من مجموعات العمليات المخصصة.

.. function:: _export(model, args, f, export_params=True, verbose=False, training=False, input_names=None, output_names=None, operator_export_Multiplier=None, opset_version=None, do_constant_folding=True, example_outputs=None, dynamic_axes=None, custom_opsets=None)

   نفس التصدير ولكن لا يتحقق من وجود ملف الإخراج.

.. function:: register_custom_op_symbolic(name، symbolic_function، symbolic_function_backward=None)

   تسجيل دالة رمزية لتصدير المشغل المخصص.

   الحجة:
      name: اسم المشغل المخصص.
      symbolic_function: الدالة الرمزية التي تحدد سلوك المشغل المخصص.
      symbolic_function_backward: الدالة الرمزية للتفاضل التلقائي للمشغل المخصص. اختياري.

.. function:: register_opset_version(name, version)

   تسجيل إصدار مجموعة العمليات لمجموعة عمليات ONNX.

   الحجة:
      name: اسم مجموعة العمليات.
      version: إصدار مجموعة العمليات.

.. function:: set_operator_export_type(operator_export_type)

   تعيين نوع التصدير.

   الحجة:
      operator_export_type: نوع التصدير. يمكن أن يكون "ONNX" أو "ONNX_ATEN".

.. function:: set_training(training)

   تعيين وضع التدريب للنموذج.

   الحجة:
      training: إذا كان النموذج في الوضع التدريبي.

.. function:: is_in_onnx_export()

   التحقق مما إذا كنا في عملية تصدير ONNX.

.. function:: handle_custom_op(ctx, name, inputs)

   التعامل مع المشغل المخصص أثناء التصدير.

   الحجة:
      ctx: سياق التصدير.
      name: اسم المشغل المخصص.
      inputs: قائمة من المدخلات للمشغل المخصص.

.. function:: register_standard_custom_ops()

   تسجيل المشغلين المخصصين القياسيين.

.. function:: register_custom_ops(custom_op_dict)

   تسجيل المشغلين المخصصين من القاموس.

   الحجة:
      custom_op_dict: قاموس من أسماء المشغلين المخصصين إلى الدوال الرمزية.

.. function:: register_custom_op(name, symbolic_function, symbolic_function_backward=None)

   تسجيل مشغل مخصص. نفس register_custom_op_symbolic.

.. function:: prepare_torch_op(ctx, torch_op, args, kwargs)

   إعداد مشغل PyTorch للتصدير.

   الحجة:
      ctx: سياق التصدير.
      torch_op: مشغل PyTorch.
      args: قائمة من المدخلات للمشغل.
      kwargs: قاموس من الكلمات الرئيسية للمشغل.

.. function:: prepare_aten_op(ctx, aten_op, args, kwargs)

   إعداد مشغل ATen للتصدير.

   الحجة:
      ctx: سياق التصدية.
      aten_op: مشغل ATen.
      args: قائمة من المدخلات للمشغل.
      kwargs: قاموس من الكلمات الرئيسية للمشغل.

.. function:: export_to_pretty_string(model, args, operator_export_type=None, opset_version=None, do_constant_folding=True, dynamic_axes=None, custom_opsets=None)

   تصدير النموذج إلى سلسلة ONNX جميلة.

   الحجة:
      model: النموذج الذي سيتم تصديره.
      args: قائمة من باقي المدخلات المطلوبة من قبل النموذج.
      operator_export_type=None: نوع التصدير. يمكن أن يكون "ONNX" أو "ONNX_ATEN".
      opset_version=None: إصدار مجموعة العمليات. إذا لم يتم توفيره، سيتم استخدام الإصدار الافتراضي.
      do_constant_folding=True: إذا كان يجب تطبيق الثنيات الثابتة.
      dynamic_axes=None: قائمة من المحاور الديناميكية لكل مدخل وإخراج.
      custom_opsets=None: مجموعة من مجموعات العمليات المخصصة.

.. function:: use_stim_exporter()

   استخدام مصدر STIM.

.. function:: use_torch_exporter()

   استخدام مصدر PyTorch.

.. currentmodule:: torch.onnx.symbolic_helper

.. function:: parse_args(inputs, kwargs)

   تحليل المدخلات والوسيطات الإضافية.

   الحجة:
      inputs: قائمة من المدخلات.
      kwargs: قاموس من الكلمات الرئيسية.

.. function:: _parse_arg(arg)

   تحليل وسيطة واحدة.

   الحجة:
===============================

.. note::
   لتصدير نموذج ONNX باستخدام TorchDynamo بدلاً من TorchScript، راجع :func:`torch.onnx.dynamo_export`.

.. contents:: :local:

مثال: تحويل AlexNet من PyTorch إلى ONNX
-------------------------------------

فيما يلي نموذج بسيط يصدر نموذج AlexNet مُدرب مسبقًا إلى ملف ONNX باسم "alexnet.onnx".
يتم تشغيل الدعوة إلى "torch.onnx.export" النموذج مرة واحدة لتتبع تنفيذه ثم تصدير
النموذج الذي تم تتبعه إلى الملف المحدد::

    import torch
    import torchvision

    dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
    model = torchvision.models.alexnet(pretrained=True).cuda()

    # تحديد أسماء الإدخال والإخراج يحدد أسماء العرض للقيم
    # داخل رسم النموذج. لا يؤدي تعيين هذه الأسماء إلى تغيير دلالية
    # الرسم؛ فهو فقط من أجل قابلية القراءة.
    #
    # تتكون إدخالات الشبكة من القائمة المسطحة للإدخالات (أي
    # القيم التي ستقوم بتمريرها إلى طريقة forward()) متبوعة بالقائمة المسطحة للمعلمات. يمكنك تحديد الأسماء جزئيًا، أي تقديم
    # قائمة هنا أقصر من عدد الإدخالات إلى النموذج، وسنقوم
    # بتعيين مجموعة فرعية فقط من الأسماء، بدءًا من البداية.
    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]

    torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)

يحتوي ملف "alexnet.onnx" الناتج على `بروتوكول ثنائي <https://developers.google.com/protocol-buffers/>`_
الذي يحتوي على كل من بنية الشبكة ومعلمات النموذج الذي تم تصديره
(في هذه الحالة، AlexNet). يؤدي تحديد "verbose=True" إلى
قيام المصدر بطباعة تمثيل قابل للقراءة بشريًا للنموذج::

    # هذه هي المدخلات والمعلمات للشبكة، والتي تحمل الأسماء التي حددناها سابقًا.
    graph(%actual_input_1 : Float(10, 3, 224, 224)
          %learned_0 : Float(64, 3, 11, 11)
          %learn0 : Float(64)
          %learned_2 : Float(192, 64, 5, 5)
          %learned_3 : Float(192)
          # ---- تم الحذف للاختصار ----
          %learned_14 : Float(1000, 4096)
          %learned_15 : Float(1000)) {
      # يتكون كل بيان من بعض المخرجات (وأنواعها)،
      # المشغل الذي سيتم تشغيله (مع سماته، مثل kernels وstrides،
      # وما إلى ذلك)، مدخلاته (%actual_input_1، %learned_0، %learned_1)
      %17 : Float(10, 64, 55, 55) = onnx::Conv[dilations=[1, 1], group=1, kernel_shape=[11, 11], pads=[2, 2, 2, 2], strides=[4, 4]](%actual_input_1, %learned_0, %learned_1), scope: AlexNet/Sequential[features]/Conv2d[0]
      %18 : Float(10, 64, 55, 55) = onnx::Relu(%17), scope: AlexNet/Sequential[features]/ReLU[1]
      %19 : Float(10, 64, 27, 27) = onnx::MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%18), scope: AlexNet/Sequential[features]/MaxPool2d[2]
      # ---- تم الحذف للاختصار ----
      %29 : Float(10, 256, 6, 6) = onnx::MaxPool[kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[2, 2]](%28), scope: AlexNet/Sequential[features]/MaxPool2d[12]
      # يعني الديناميكي أن الشكل غير معروف. قد يكون ذلك بسبب
      # حد في تنفيذنا (والذي نود إصلاحه في
      # إصدار مستقبلي) أو الأشكال التي تكون ديناميكية حقًا.
      %30 : Dynamic = onnx::Shape(%29), scope: AlexNet
      %31 : Dynamic = onnx::Slice[axes=[0], ends=[1], starts=[0]](%30), scope: AlexNet
      %32 : Long() = onnx::Squeeze[axes=[0]](%31), scope: AlexNet
      %33 : Long() = onnx::Constant[value={9216}](), scope: AlexNet
      # ---- تم الحذف للاختصار ----
      %output1 : Float(10, 1000) = onnx::Gemm[alpha=1, beta=1, broadcast=1, transB=1](%45, %learned_14, %learned_15), scope: AlexNet/Sequential[classifier]/Linear[6]
      return (%output1);
    }

يمكنك أيضًا التحقق من الإخراج باستخدام مكتبة `ONNX <https://github.com/onnx/onnx/>`_،
والتي يمكنك تثبيتها باستخدام ``pip``::

    pip install onnx

بعد ذلك، يمكنك تشغيل::

    import onnx

    # تحميل نموذج ONNX
    model = onnx.load("alexnet.onnx")

    # التحقق من أن النموذج جيد التشكيل
    onnx.checker.check_model(model)

    # طباعة تمثيل قابل للقراءة بشريًا للرسم البياني
    print(onnx.helper.printable_graph(model.graph))

يمكنك أيضًا تشغيل النموذج المصدر باستخدام إحدى بيئات التشغيل العديدة
`التي تدعم ONNX <https://onnx.ai/supported-tools.html#deployModel>`_.
على سبيل المثال، بعد تثبيت `ONNX Runtime <https://www.onnxruntime.ai>`_، يمكنك
تحميل النموذج وتشغيله::

    import onnxruntime as ort
    import numpy as np

    ort_session = ort.InferenceSession("alexnet.onnx")

    outputs = ort_session.run(
        None,
        {"actual_input_1": np.random.randn(10, 3, 224, 224).astype(np.float32)},
    )
    print(outputs[0])

هذا `دليل أكثر تفصيلاً حول تصدير نموذج وتشغيله باستخدام ONNX Runtime <https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html>`_.

.. _tracing-vs-scripting:

التتبع مقابل البرمجة النصية
--------------------

داخليًا، يتطلب :func:`torch.onnx.export()` :class:`torch.jit.ScriptModule` بدلاً من
:class:`torch.nn.Module`. إذا لم يكن النموذج الذي تم تمريره بالفعل عبارة عن ``ScriptModule``،
فسيستخدم "export()" *التتبع* لتحويله إلى واحد:

.. TODO(justinchuby): إضافة كلمة حول التوصية بالتتبع على البرمجة النصية لمعظم حالات الاستخدام.

* **التتبع**: إذا تم استدعاء "torch.onnx.export()" باستخدام نموذج Module ليس بالفعل
  "ScriptModule"، فإنه يقوم أولاً بما يعادل :func:`torch.jit.trace`، والذي ينفذ النموذج
  مرة واحدة مع "args" المحددة ويسجل جميع العمليات التي تحدث أثناء هذا التنفيذ. وهذا
  يعني أنه إذا كان نموذجك ديناميكيًا، أي يغير سلوكه اعتمادًا على بيانات الإدخال، فلن يقوم النموذج المصدر
  بالتقاط هذا السلوك الديناميكي.
  نوصي بفحص النموذج المصدر والتأكد من أن المشغلين يبدون
  معقولون. سيقوم التتبع بإلغاء حلقات التكرار والعبارات الشرطية، ويصدر رسمًا بيانيًا ثابتًا مطابقًا تمامًا
  مثل الجريان الذي تم تتبعه. إذا كنت تريد تصدير نموذجك مع تدفق تحكم ديناميكي، فستحتاج
  إلى استخدام *البرمجة النصية*.

* **البرمجة النصية**: يحافظ التجميع عبر البرمجة النصية على تدفق التحكم الديناميكي وهو صالح للإدخالات
  ذات الأحجام المختلفة. لاستخدام البرمجة النصية:

  * استخدم :func:`torch.jit.script` لإنتاج "ScriptModule".
  * استدعاء "torch.onnx.export()" مع "ScriptModule" كنموذج. لا تزال "args" مطلوبة،
    ولكن سيتم استخدامها داخليًا فقط لإنتاج إخراج مثال، بحيث يمكن التقاط أنواع وأشكال
    الإخراجات. لن يتم إجراء أي تتبّع.

راجع `مقدمة إلى TorchScript <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html>`_
و `TorchScript <jit.html>`_ لمزيد من التفاصيل، بما في ذلك كيفية الجمع بين التتبع والبرمجة النصية لتلبية
المتطلبات الخاصة لموديلات مختلفة.

تجنب المشاكل
-----------------

تجنب NumPy وأنواع Python المضمنة
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

يمكن كتابة نماذج PyTorch باستخدام NumPy أو أنواع ووظائف Python، ولكن
أثناء :ref:`tracing<tracing-vs-scripting>`، يتم تحويل أي متغيرات من أنواع NumPy أو Python
(بدلاً من torch.Tensor) إلى ثوابت، والتي ستنتج
النتيجة الخاطئة إذا كان ينبغي أن تتغير هذه القيم اعتمادًا على الإدخالات.

على سبيل المثال، بدلاً من استخدام وظائف numpy على numpy.ndarrays: ::

    # سيئ! سيتم استبداله بثوابت أثناء التتبع.
    x, y = np.random.rand(1, 2), np.random.rand(1, 2)
    np.concatenate((x, y), axis=1)

استخدم مشغلات torch على torch.Tensors: ::

    # جيد! سيتم التقاط عمليات Tensor أثناء التتبع.
    x, y = torch.randn(1, 2), torch.randn(1, 2)
    torch.cat((x, y), dim=1)


وبدلاً من استخدام :func:`torch.Tensor.item` (الذي يحول Tensor إلى رقم Python
مضمن): ::

    # سيئ! سيتم استبدال y.item() بثابت أثناء التتبع.
    def forward(self, x, y):
        return x.reshape(y.item(), -1)

استخدم دعم PyTorch للتحويل الضمني لtensors ذات العنصر الواحد: ::

    # جيد! سيتم الاحتفاظ بـ y كمتغير أثناء التتبع.
    def forward(self, x, y):
        return x.reshape(y, -1)

تجنب Tensor.data
^^^^^^^^^^^^^^^^^

يمكن أن يؤدي استخدام حقل Tensor.data إلى إنتاج تتبع غير صحيح وبالتالي رسم بياني ONNX غير صحيح.
استخدم :func:`torch.Tensor.detach` بدلاً من ذلك. (العمل جارٍ على
`إزالة Tensor.data تمامًا <https://github.com/pytorch/pytorch/issues/30987>`_).

تجنب العمليات في الموقع عند استخدام tensor.shape في وضع التتبع
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

في وضع التتبع، يتم تتبع الأشكال التي تم الحصول عليها من "tensor.shape" كtensors،
وتشارك نفس الذاكرة. قد يتسبب ذلك في عدم تطابق قيم الإخراج النهائية.
كحل بديل، تجنب استخدام العمليات في الموقع في هذه السيناريوهات.
على سبيل المثال، في النموذج::

    class Model(torch.nn.Module):
      def forward(self, states):
          batch_size, seq_length = states.shape[:2]
          real_seq_length = seq_length
          real_seq_length += 2
          return real_seq_length + seq_length

يشترك "real_seq_length" و "seq_length" في نفس الذاكرة في وضع التتبع.
يمكن تجنب ذلك عن طريق إعادة كتابة العملية في الموقع::

    real_seq_length = real_seq_length + 2

القيود
بالتأكيد! فيما يلي النص المترجم بتنسيق ReStructuredText:

-----------

الأنواع
^^^^^^^

* يدعم كمدخلات أو مخرجات للنموذج فقط :class: `torch.Tensors`، والأنواع الرقمية التي يمكن تحويلها بسهولة إلى :class: `torch.Tensors` (مثل float، int)، والمجاميع والقوائم التي تحتوي على تلك الأنواع. وتُقبل المدخلات والمخرجات من نوع dict و str في وضع :ref: `tracing<tracing-vs-scripting>`، ولكن:

  * سيتم استبدال أي عملية حسابية تعتمد على قيمة إدخال من نوع dict أو str **بقيمة ثابتة** تم ملاحظتها أثناء التنفيذ الذي تم تتبعه.
  * سيتم استبدال أي مخرج من نوع dict **بتسلسل مسطح لقيمه** (سيتم إزالة المفاتيح) بشكل صامت. على سبيل المثال، يتم تحويل ``{"foo": 1، "bar": 2}`` إلى ``(1، 2)``.
  * سيتم إزالة أي مخرج من نوع str بشكل صامت.

* لا يتم دعم بعض العمليات التي تتضمن المجاميع والقوائم في وضع :ref: `scripting<tracing-vs-scripting>` بسبب الدعم المحدود في ONNX للتسلسلات المُعششة. وعلى وجه التحديد، لا يتم دعم إضافة مجموعة إلى قائمة. في وضع التتبع، يتم تسطيح التسلسلات المُعششة تلقائيًا أثناء التتبع.

الاختلافات في تنفيذ المشغل
^^^^^^^^^^^^^^^^^^^^^^^^^^

بسبب الاختلافات في تنفيذ المشغل، فقد يؤدي تشغيل النموذج المصدر على ركائز مختلفة إلى نتائج مختلفة عن بعضها البعض أو عن PyTorch. عادة ما تكون هذه الاختلافات صغيرة من الناحية الرقمية، لذلك يجب ألا يكون هذا مصدر قلق إلا إذا كانت تطبيقاتك حساسة لهذه الاختلافات الصغيرة.

.. _tensor-indexing:

أنماط فهرسة Tensor غير المدعومة
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

فيما يلي أنماط فهرسة Tensor التي لا يمكن تصديرها.
إذا كنت تواجه مشكلات في تصدير نموذج لا يتضمن أيًا من الأنماط غير المدعومة أدناه، يرجى التحقق المزدوج من أنك تقوم بالتصدير باستخدام أحدث إصدار من ``opset_version``.

القراءات / الجلب
~~~~~~~~~~~~~~~

عند الفهرسة في Tensor للقراءة، لا يتم دعم الأنماط التالية: ::

  # مؤشرات Tensor التي تشمل قيمًا سلبية.
  data[torch.tensor([[1, 2], [2, -3]]), torch.tensor([-2, 3])]
  # الحلول البديلة: استخدم قيم المؤشرات الإيجابية.

الكتابات / الإعدادات
~~~~~~~~~~~~~~~~~~~

عند الفهرسة في Tensor للكتابة، لا يتم دعم الأنماط التالية: ::

  # مؤشرات Tensor متعددة إذا كان أي منها يحمل مرتبة >= 2
  data[torch.tensor([[1, 2], [2, 3]]), torch.tensor([2, 3])] = new_data
  # الحلول البديلة: استخدم مؤشر Tensor واحد يحمل مرتبة >= 2،
  # أو استخدم مؤشرات Tensor متعددة متتالية تحمل مرتبة == 1.

  # مؤشرات Tensor متعددة غير متتالية
  data[torch.tensor([2, 3]), :, torch.tensor([1, 2])] = new_data
  # الحلول البديلة: قم بترحيل `data` بحيث تكون مؤشرات Tensor متتالية.

  # مؤشرات Tensor التي تشمل قيمًا سلبية.
  data[torch.tensor([1, -2]), torch.tensor([-2, 3])] = new_data
  # الحلول البديلة: استخدم قيم المؤشرات الإيجابية.

  # البث الضمني المطلوب لـ new_data.
  data[torch.tensor([[0, 2], [1, 1]]), 1:3] = new_data
  # الحلول البديلة: قم بتوسيع new_data بشكل صريح.
  # مثال:
  #   شكل data: [3، 4، 5]
  #   شكل new_data: [5]
  #   الشكل المتوقع لـ new_data بعد البث: [2، 2، 2، 5]

إضافة الدعم للمشغلين
عندما تقوم بتصدير نموذج يتضمن مشغلات غير مدعومة، فسوف ترى رسالة خطأ مشابهة لما يلي:

.. code-block:: text

    RuntimeError: فشل التصدير ONNX: لم يتم التمكن من تصدير المشغل foo

عندما يحدث ذلك، هناك بعض الأمور التي يمكنك القيام بها:

1. قم بتغيير النموذج بحيث لا يستخدم هذا المشغل.
2. قم بإنشاء دالة رمزية لتحويل المشغل وقم بتسجيلها كدالة رمزية مخصصة.
3. ساهم في PyTorch بإضافة نفس الدالة الرمزية إلى :mod:`torch.onnx` نفسها.

إذا قررت تنفيذ دالة رمزية (نأمل أن تساهم بها مرة أخرى في PyTorch!)، فهذا هو ما يمكنك البدء به:

تفاصيل مصدر ONNX الداخلي
^^^^^^^^^^^^^^^^^^^^^^^^^^

الدالة "الرمزية" هي دالة تقوم بتفكيك مشغل PyTorch إلى تركيبة من سلسلة من مشغلات ONNX.

أثناء التصدير، تتم زيارة كل عقدة (والتي تحتوي على مشغل PyTorch) في الرسم البياني TorchScript
من قبل المصدر بالترتيب التصاعدي.
عند زيارة عقدة، يبحث المصدر عن دالات رمزية مسجلة لذلك المشغل. يتم تنفيذ الدالات الرمزية في Python.
سوف تبدو دالة رمزية لمشغل باسم ``foo`` على الشكل التالي::

    def foo(
      g,
      input_0: torch._C.Value,
      input_1: torch._C.Value) -> Union[None, torch._C.Value, List[torch._C.Value]]:
      """
      تضيف عمليات ONNX التي تمثل دالة PyTorch هذه عن طريق تحديث
      الرسم البياني g باستخدام مكالمات `g.op()`.

      Args:
        g (Graph): الرسم البياني لكتابة التمثيل ONNX فيه.
        input_0 (Value): القيمة التي تمثل المتغيرات التي تحتوي على
            الإدخال الأول لهذا المشغل.
        input_1 (Value): القيمة التي تمثل المتغيرات التي تحتوي على
            الإدخال الثاني لهذا المشغل.

      Returns:
        قيمة أو قائمة من القيم التي تحدد عقد ONNX التي تحسب شيئًا
        مكافئًا لمشغل PyTorch الأصلي مع الإدخالات المعطاة.

        لا شيء إذا لم يمكن تحويله إلى ONNX.
      """
      ...

أنواع ``torch._C`` هي أغلفة Python حول الأنواع المحددة في C++ في
`ir.h <https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/ir/ir.h>`_.

تعتمد عملية إضافة دالة رمزية على نوع المشغل.

.. _adding-support-aten:

مشغلات ATen
^^^^^^^^^^^^^^

`ATen <https://pytorch.org/cppdocs/#aten>`_ هي مكتبة التنسور المدمجة في PyTorch.
إذا كان المشغل هو مشغل ATen (يظهر في الرسم البياني TorchScript بالبادئة
``aten::``)، فتأكد من أنه غير مدعوم بالفعل.

قائمة المشغلات المدعومة
~~~~~~~~~~~~~~~~~~~~~~~~~~~

قم بزيارة :doc:`قائمة مشغلات TorchScript المدعومة <../onnx_torchscript_supported_aten_ops>`
التي تم إنشاؤها تلقائيًا للحصول على تفاصيل حول المشغل المدعوم في كل ``opset_version``.

إضافة الدعم لمشغل aten أو كمي
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

إذا لم يكن المشغل في القائمة أعلاه:

* قم بتعريف الدالة الرمزية في ``torch/onnx/symbolic_opset<version>.py``، على سبيل المثال
  `torch/onnx/symbolic_opset9.py <https://github.com/pytorch/pytorch/blob/main/torch/onnx/symbolic_opset9.py>`_.
  تأكد من أن للدالة نفس الاسم مثل دالة ATen، والتي قد يتم إعلانها في
  ``torch/_C/_VariableFunctions.pyi`` أو ``torch/nn/functional.pyi`` (هذه الملفات يتم إنشاؤها في
  وقت البناء، لذلك لن تظهر في عملية الفحص حتى تقوم ببناء PyTorch).
* بشكل افتراضي، يكون الحجة الأولى هي الرسم البياني ONNX.
  يجب أن تتطابق أسماء الحجج الأخرى تمامًا مع الأسماء في ملف ``.pyi``،
  لأن الإرسال يتم باستخدام وسيطات الكلمات الرئيسية.
* في الدالة الرمزية، إذا كان المشغل في
  `مجموعة مشغلات ONNX القياسية <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_،
  فنحن بحاجة فقط إلى إنشاء عقدة لتمثيل مشغل ONNX في الرسم البياني.
  إذا لم يكن الأمر كذلك، فيمكننا تركيب العديد من المشغلات القياسية التي لها
  الدلالات المكافئة لمشغل ATen.

هذا مثال على التعامل مع دالة رمزية مفقودة لمشغل ``ELU``.

إذا قمنا بتشغيل الشفرة التالية::

    print(
        torch.jit.trace(
            torch.nn.ELU(), # module
            torch.ones(1)   # example input
        ).graph
    )

سنرى شيئًا مثل::

  graph(%self : __torch__.torch.nn.modules.activation.___torch_mangle_0.ELU,
        %input : Float(1, strides=[1], requires_grad=0, device=cpu)):
    %4 : float = prim::Constant[value=1.]()
    %5 : int = prim::Constant[value=1]()
    %6 : int = prim::Constant[value=1]()
    %7 : Float(1, strides=[1], requires_grad=0, device=cpu) = aten::elu(%input, %4, %5, %6)
    return (%7)

نظرًا لأننا نرى ``aten::elu`` في الرسم البياني، فنحن نعلم أن هذا هو مشغل ATen.

نتحقق من `قائمة مشغلات ONNX <https://github.com/onnx/onnx/blob/master/docs/Operators.md>`_،
ونؤكد أن ``Elu`` معياري في ONNX.

نجد توقيعًا لـ ``elu`` في ``torch/nn/functional.pyi``::

    def elu(input: Tensor, alpha: float = ..., inplace: bool = ...) -> Tensor: ...

نضيف السطور التالية إلى ``symbolic_opset9.py``::

    def elu(g, input: torch.Value, alpha: torch.Value, inplace: bool = False):
        return g.op("Elu", input, alpha_f=alpha)

الآن، يمكن لـ PyTorch تصدير النماذج التي تحتوي على مشغل ``aten::elu``!

راجع ملفات ``torch/onnx/symbolic_opset*.py`` للحصول على المزيد من الأمثلة.


torch.autograd.Functions
^^^^^^^^^^^^^^^^^^^^^^^^

إذا كان المشغل هو فئة فرعية من :class:`torch.autograd.Function`، فهناك ثلاث طرق
لتصديره.

الطريقة الرمزية الثابتة
~~~~~~~~~~~~~~~~~~~~~~

يمكنك إضافة طريقة ثابتة باسم ``symbolic`` إلى فئة الدالة الخاصة بك. يجب أن تقوم بإرجاع
مشغلات ONNX التي تمثل سلوك الدالة في ONNX. على سبيل المثال::

    class MyRelu(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input: torch.Tensor) -> torch.Tensor:
            ctx.save_for_backward(input)
            return input.clamp(min=0)

        @staticmethod
        def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
            return g.op("Clip", input, g.op("Constant", value_t=torch.tensor(0, dtype=torch.float)))

.. FIXME(justinchuby): PythonOps معقدة للغاية والمثال أدناه
..  يستخدم طرق خاصة لا نقوم بتعريضها. نحن نبحث لتحسين التجربة.
..  نظرًا لأن SymbolicContext قديمة، فإننا نعتقد
..  أن تعريف طريقة ثابتة رمزية هو طريقة أفضل للذهاب الآن.

.. PythonOp رمزي
.. ~~~~~~~~~~~~~~~~~

.. كبديل، يمكنك تسجيل دالة رمزية مخصصة.
.. يمنح هذا الدالة الرمزية إمكانية الوصول إلى المزيد من المعلومات من خلال كائن
.. ``torch.onnx.SymbolicContext``، والذي يتم تمريره كأول
.. حجة (قبل كائن ``Graph``).

.. تظهر جميع دالات ``Function`` في الرسم البياني TorchScript كعقد ``prim::PythonOp``.
.. للتمييز بين فئات ``Function`` الفرعية المختلفة،
.. يجب أن تستخدم الدالة الرمزية وسيط kwarg الذي يتم تعيينه باسم الفئة.

.. يجب أن تضيف الدالات الرمزية المخصصة معلومات النوع والشكل عن طريق استدعاء ``setType(...)``
.. على كائنات Value قبل إرجاعها (تم تنفيذها في C++ بواسطة
.. . ``torch::jit::Value::setType``). هذا ليس مطلوبًا، ولكنه يمكن أن يساعد استدلال الشكل والنوع المصدر
.. للعقد الموجودة في الأسفل. لمثال غير بسيط على ``setType``، راجع
.. ``test_aten_embedding_2`` في
.. `test_operators.py <https://github.com/pytorch/pytorch/blob/main/test/onnx/test_operators.py>`_.

.. يُظهر المثال أدناه كيفية الوصول إلى ``requires_grad`` عبر كائن ``Node``:

..     class MyClip(torch.autograd.Function):
..         @staticmethod
..         def forward(ctx, input, min):
..             ctx.save_for_backward(input)
..             return input.clamp(min=min)

..     class MyRelu(torch.autograd.Function):
..         @staticmethod
..         def forward(ctx, input):
..             ctx.save_for_backward(input)
..             return input.clamp(min=0)

..     def symbolic_python_op(g: "GraphContext", *args, **kwargs):
..         n = ctx.cur_node
..         print("original node: ", n)
..         for i, out in enumerate(n.outputs()):
..             print("original output {}: {}, requires grad: {}".format(i, out, out.requiresGrad()))
..         import torch.onnx.symbolic_helper as sym_helper
..         for i, arg in enumerate(args):
..             requires_grad = arg.requiresGrad() if sym_helper._is_value(arg) else False
..             print("arg {}: {}, requires grad: {}".format(i, arg, requires_grad))

..         name = kwargs["name"]
..         ret = None
..         if name == "MyClip":
..             ret = g.op("Clip", args[0], args[1])
..         elif name == "MyRelu":
..             ret = g.op("Relu", args[0])
..         else:
..             # يسجل تحذيرًا ويعيد None
..             return _unimplemented("prim::PythonOp"، "نوع العقدة غير معروف: " + name)
..         # نسخ النوع والشكل من العقدة الأصلية.
..         ret.setType(n.type())
..         return ret

..     from torch.onnx import register_custom_op_symbolic
.. .     register_custom_op_symbolic("prim::PythonOp"، symbolic_python_op، 1)

دالة Autograd المضمنة
~~~~~~~~~~~~~~~~~~~~~~~~

في الحالات التي لا يتم فيها توفير طريقة رمزية ثابتة لـ :class:`torch.autograd.Function` اللاحقة أو
حيث لا يتم توفير دالة لتسجيل ``prim::PythonOp`` كدالات رمزية مخصصة،
يحاول :func:`torch.onnx.export` تضمين الرسم البياني الذي يقابل ذلك :class:`torch.autograd.Function` بحيث
يتم تقسيم هذه الدالة إلى مشغلات فردية تم استخدامها داخل الدالة.
يجب أن يكون التصدير ناجحًا طالما أن هذه المشغلات الفردية مدعومة. على سبيل المثال::

    class MyLogExp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input: torch.Tensor) -> torch.Tensor:
            ctx.save_for_backward(input)
            h = input.exp()
            return h.log().log()

لا توجد طريقة رمزية ثابتة موجودة لهذا النموذج، ومع ذلك يتم تصديره على النحو التالي::

    graph(%input : Float(1, strides=[1], requires_grad=0, device=cpu)):
        %1 : float = onnx::Exp[](%input)
        %2 : float = onnx::Log[](%1)
        %3 : float = onnx::Log[](%2)
        return (%3)

إذا كنت بحاجة إلى تجنب تضمين :class:`torch.autograd.Function`، فيجب عليك تصدير النماذج باستخدام
``operator_export_type`` المحدد إلى ``ONNX_FALLTHROUGH`` أو ``ONNX_ATEN_FALLBACK``.

مشغلات مخصصة
^^^^^^^^^^^^^^^^

يمكنك تصدير نموذجك بمشغلات مخصصة تتضمن مزيجًا من العديد من مشغلات ONNX القياسية،
أو يتم تشغيلها بواسطة backend C++ ذاتي التعريف.

دالات ONNX-script
~~~~~~~~~~~~~~~~~~~~~

إذا كان المشغل ليس مشغل ONNX قياسي، ولكنه يمكن أن يتكون من عدة مشغلات ONNX موجودة، فيمكنك الاستفادة من
`ONNX-script <https://github.com/microsoft/onnx-script>`_ لإنشاء دالة ONNX خارجية لدعم المشغل.
يمكنك تصديره باتباع هذا المثال::

    import onnxscript
    # هناك ثلاث إصدارات لمجموعة المشغلات مطلوبة للمحاذاة
    # هذا هو (1) إصدار مجموعة المشغلات في دالة ONNX
    from onnxscript.onnx_opset import opset15 as op
    opset_version = 15

    x = torch.randn(1, 2, 3, 4, requires_grad=True)
    model = torch.nn.SELU()

    custom_opset = onnxscript.values.Opset(domain="onnx-script"، version=1)

    @onnxscript.script(custom_opset)
    def Selu(X):
        alpha = 1.67326  # يتم لفها تلقائيًا كثوابت
        gamma = 1.0507
        alphaX = op.CastLike(alpha, X)
        gammaX = op.CastLike(gamma, X)
        neg = gammaX * (alphaX * op.Exp(X) - alphaX)
        pos = gammaX * X
        zero = op.CastLike(0, X)
        return op.Where(X <= zero, neg, pos)

    # توفر واجهة برمجة التطبيقات setType معلومات الشكل/النوع لاستدلال الشكل/النوع ONNX
    def custom_selu(g: jit_utils.GraphContext, X):
        return g.onnxscript_op(Selu, X).setType(X.type())

    # تسجيل الدالة الرمزية المخصصة
    # هناك ثلاث إصدارات لمجموعة المشغلات مطلوبة للمحاذاة
    # هذا هو (2) إصدار مجموعة المشغلات في السجل
    torch.onnx.register_custom_op_symbolic(
        symbolic_name="aten::selu"،
        symbolic_fn=custom_selu،
        opset_version=opset_version،
    )

    # هناك ثلاث إصدارات لمجموعة المشغلات مطلوبة للمحاذاة
    # هذا هو (2) إصدار مجموعة المشغلات في المصدر
    torch.onnx.export(
        model،
        x،
        "model.onnx"،
        opset_version=opset_version،
        # مطلوب فقط إذا كنت تريد تحديد إصدار مجموعة مشغلات > 1.
        custom_opsets={"onnx-script": 2}
    )
هذا مثال على كيفية تصدير مشغل مخصص في مجموعة أوامر "onnx-script".

عند تصدير مشغل مخصص، يمكنك تحديد إصدار مجال مخصص باستخدام قاموس "custom_opsets" أثناء التصدير. إذا لم يتم تحديده، فإن إصدار مجموعة أوامر المشغل المخصص يكون 1 بشكل افتراضي.

ملاحظة: كن حذرًا في محاذاة إصدار مجموعة الأوامر المذكورة في المثال أعلاه، وتأكد من استخدامها في خطوة المصدر. يعد مثال الاستخدام حول كيفية كتابة دالة "onnx-script" إصدارًا تجريبيًا من حيث التطوير النشط لـ "onnx-script". يرجى اتباع أحدث إصدارات "ONNX-script" على GitHub.

مشغلات C++
~~~~~~~~~~

إذا كان النموذج يستخدم مشغل مخصصًا منفذًا في C++ كما هو موضح في "توسيع TorchScript بمشغلات C++ المخصصة"، فيمكنك تصديره باتباع هذا المثال::

    from torch.onnx import symbolic_helper


    # Define custom symbolic function
    @symbolic_helper.parse_args("v", "v", "f", "i")
    def symbolic_foo_forward(g, input1, input2, attr1, attr2):
        return g.op("custom_domain::Foo", input1, input2, attr1_f=attr1, attr2_i=attr2)


    # Register custom symbolic function
    torch.onnx.register_custom_op_symbolic("custom_ops::foo_forward", symbolic_foo_forward, 9)


    class FooModel(torch.nn.Module):
        def __init__(self, attr1, attr2):
            super().__init__()
            self.attr1 = attr1
            self.attr2 = attr2

        def forward(self, input1, input2):
            # Calling custom op
            return torch.ops.custom_ops.foo_forward(input1, input2, self.attr1, self.attr2)


    model = FooModel(attr1, attr2)
    torch.onnx.export(
        model,
        (example_input1, example_input1),
        "model.onnx",
        # only needed if you want to specify an opset version > 1
        custom_opsets={"custom_domain": 2}
    )

يقوم المثال أعلاه بتصديره كمشغل مخصص في مجموعة أوامر "custom_domain".

عند تصدير مشغل مخصص، يمكنك تحديد إصدار المجال المخصص باستخدام قاموس "custom_opsets" أثناء التصدير. إذا لم يتم تحديده، فإن إصدار مجموعة أوامر المشغل المخصص يكون 1 بشكل افتراضي.

يجب أن يدعم وقت التشغيل الذي يستخدم النموذج المشغل المخصص. راجع "مشغلات Caffe2 المخصصة" أو "مشغلات ONNX Runtime المخصصة"، أو وثائق وقت تشغيل اختيارك للحصول على التفاصيل.

اكتشاف جميع عمليات ATen غير القابلة للتحويل مرة واحدة
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

عندما يفشل التصدير بسبب عملية ATen غير قابلة للتحويل، فقد يكون هناك في الواقع أكثر من عملية واحدة ولكن رسالة الخطأ تذكر فقط الأولى. لاكتشاف جميع العمليات غير القابلة للتحويل في مرة واحدة، يمكنك القيام بما يلي::

    # prepare model, args, opset_version
    ...

    torch_script_graph, unconvertible_ops = torch.onnx.utils.unconvertible_ops(
        model, args, opset_version=opset_version
    )

    print(set(unconvertible_ops))

المجموعة تقريبية لأن بعض العمليات قد تتم إزالتها أثناء عملية التحويل ولا تحتاج إلى تحويل. وقد يكون لبعض العمليات الأخرى دعم جزئي قد يفشل التحويل مع إدخالات معينة، ولكن هذا يجب أن يعطيك فكرة عامة عن العمليات غير المدعومة. لا تتردد في فتح قضايا GitHub لطلبات دعم العمليات.

الأسئلة الشائعة
.. _أسئلة متكررة حول ONNX:

أسئلة متكررة حول ONNX
--------------------------

س: لقد قمت بتصدير نموذج LSTM الخاص بي، ولكن يبدو أن حجم المدخلات ثابت؟

  يقوم الـ tracer بتسجيل أشكال المدخلات المثالية. إذا كان يجب على النموذج قبول مدخلات ذات أشكال ديناميكية، قم بضبط ``dynamic_axes`` عند استدعاء :func:`torch.onnx.export`.

س: كيف يمكن تصدير النماذج التي تحتوي على حلقات؟

  راجع `التتبع مقابل البرمجة النصية`_.

س: كيف يمكن تصدير النماذج ذات المدخلات من النوع الأساسي (مثل int، float)؟

  تمت إضافة دعم المدخلات من الأنواع الرقمية الأساسية في PyTorch 1.9.
  ومع ذلك، لا يدعم المصدر التصدير النماذج ذات المدخلات النصية (str).

س: هل يدعم ONNX التحويل الضمني لنوع البيانات القياسي؟

  لا تدعم مواصفات ONNX ذلك، ولكن سيحاول المصدر التعامل مع هذا الجزء.
  يتم تصدير القيم القياسية على أنها مصفوفات ثابتة.
  سيحدد المصدر نوع البيانات الصحيح للقيم القياسية. في الحالات النادرة التي لا يتمكن فيها من ذلك، سيتعين عليك تحديد نوع البيانات يدويًا باستخدام ``dtype=torch.float32`` على سبيل المثال.
  إذا رأيت أي أخطاء، يرجى `إنشاء قضية على GitHub <https://github.com/pytorch/pytorch/issues>`_.

س: هل يمكن تصدير القوائم من Tensor إلى ONNX؟

  نعم، بالنسبة لـ ``opset_version`` >= 11، حيث تم تقديم ONNX لنوع Sequence في opset 11.

واجهة برمجة التطبيقات الخاصة بـ Python
----------

.. automodule:: torch.onnx

الدوال
^^^^^^^^^

.. autofunction:: export
.. autofunction:: export_to_pretty_string
.. autofunction:: register_custom_op_symbolic
.. autofunction:: unregister_custom_op_symbolic
.. autofunction:: select_model_mode_for_export
.. autofunction:: is_in_onnx_export
.. autofunction:: enable_log
.. autofunction:: disable_log
.. autofunction:: torch.onnx.verification.find_mismatch

الصفوف
^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    JitScalarType
    verification.GraphInfo
    verification.VerificationOptions