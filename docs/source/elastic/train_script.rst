.. _elastic_train_script:

نص البرنامج النصي للتدريب
--------------------

إذا كان نص البرنامج النصي للتدريب يعمل مع ``torch.distributed.launch``، فسيستمر في العمل مع ``torchrun`` مع هذه الاختلافات:

1. لا حاجة إلى تمرير ``RANK`` و ``WORLD_SIZE`` و ``MASTER_ADDR`` و ``MASTER_PORT`` يدويًا.

2. يمكن توفير ``rdzv_backend`` و ``rdzv_endpoint``. بالنسبة لمعظم المستخدمين، سيتم تعيين هذا إلى ``c10d`` (راجع `rendezvous <rendezvous.html>`_). يقوم "خيار الرجوع الافتراضي" بإنشاء نقطة التقاء غير مرنة حيث يحتوي ``rdzv_endpoint`` على عنوان الرئيسي.

3. تأكد من وجود منطق ``load_checkpoint(path)`` و ``save_checkpoint(path)`` في نص البرنامج النصي الخاص بك. عند فشل أي عدد من العمال، سنعيد تشغيل جميع العمال بنفس الحجج البرنامجية، لذلك ستفقد التقدم المحرز حتى نقطة التفتيش الأخيرة (راجع `الإطلاق المرن <run.html>`_).

4. تمت إزالة علم ``use_env``. إذا كنت تقوم بتحليل الترتيب المحلي عن طريق تحليل خيار ``--local-rank``، فيجب عليك الحصول على الترتيب المحلي من متغير البيئة ``LOCAL_RANK`` (على سبيل المثال، ``int(os.environ["LOCAL_RANK"])``).

فيما يلي مثال توضيحي لنص برنامج تدريبي يقوم بإنشاء نقطة تفتيش في كل حقبة، وبالتالي فإن أسوأ خسارة في التقدم في حالة الفشل هي قيمة حقبة كاملة من التدريب.

.. code-block:: python

  def main():
       args = parse_args(sys.argv[1:])
       state = load_checkpoint(args.checkpoint_path)
       initialize(state)

       # torch.distributed.run يضمن أن هذا سيعمل
       # عن طريق تصدير جميع متغيرات البيئة اللازمة لتهيئة مجموعة العمليات
       torch.distributed.init_process_group(backend=args.backend)

       for i in range(state.epoch, state.total_num_epochs)
            for batch in iter(state.dataset)
                train(batch, state.model)

            state.epoch += 1
            save_checkpoint(state)

للحصول على أمثلة ملموسة لنصوص البرامج النصية المطابقة لـ torchelastic، تفضل بزيارة صفحة `أمثلة <examples.html>`_ الخاصة بنا.