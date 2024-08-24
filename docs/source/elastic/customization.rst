التخصيص
========

يصف هذا القسم كيفية تخصيص TorchElastic ليناسب احتياجاتك.

مطلق التطبيق
-----------

من المفترض أن يكون برنامج الإطلاق المرفق مع TorchElastic كافيًا لمعظم حالات الاستخدام (راجع :ref: `launcher-api`).
يمكنك تنفيذ مطلق تطبيق مخصص من خلال إنشاء وكيل برمجيًا وإمرار مواصفات له للعمال كما هو موضح أدناه.

.. code-block:: python

  # my_launcher.py

  if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    rdzv_handler = RendezvousHandler(...)
    spec = WorkerSpec(
        local_world_size=args.nproc_per_node,
        fn=trainer_entrypoint_fn,
        args=(trainer_entrypoint_fn args.fn_args,...),
        rdzv_handler=rdzv_handler,
        max_restarts=args.max_restarts,
        monitor_interval=args.monitor_interval,
    )

    agent = LocalElasticAgent(spec, start_method="spawn")
    try:
        run_result = agent.run()
        if run_result.is_failed():
            print(f"فشل العامل 0 مع: run_result.failures[0]")
        else:
            print(f"قيمة الإرجاع للعامل 0 هي: run_result.return_values[0]")
    except Exception ex:
        # التعامل مع الاستثناء

معالج الالتقاء
------------------------

لتنفيذ الالتقاء الخاص بك، قم بتوسيع ``torch.distributed.elastic.rendezvous.RendezvousHandler``
وتنفيذ طرقه.

.. warning:: معالجات الالتقاء معقدة في التنفيذ. قبل أن تبدأ، تأكد من أنك تفهم تمامًا خصائص الالتقاء.
          يرجى الرجوع إلى :ref:`rendezvous-api` لمزيد من المعلومات.

بمجرد التنفيذ، يمكنك تمرير معالج الالتقاء المخصص إلى مواصفات العامل عند إنشاء الوكيل.

.. code-block:: python

    spec = WorkerSpec(
        rdzv_handler=MyRendezvousHandler(params),
        ...
    )
    elastic_agent = LocalElasticAgent(spec, start_method=start_method)
    elastic_agent.run(spec.role)

معالج القياس
-----------------------------

يقوم TorchElastic بإصدار قياسات على مستوى المنصة (راجع :ref: `metrics-api`).
وبشكل افتراضي، يتم إصدار القياسات إلى `/dev/null` لذلك لن تراها.
لإرسال القياسات إلى خدمة التعامل مع القياسات في البنية التحتية الخاصة بك،
قم بتنفيذ `torch.distributed.elastic.metrics.MetricHandler` و`configure` في مطلق التطبيق المخصص الخاص بك.

.. code-block:: python

  # my_launcher.py

  import torch.distributed.elastic.metrics as metrics

  class MyMetricHandler(metrics.MetricHandler):
      def emit(self, metric_data: metrics.MetricData):
          # إرسال metric_data إلى مصدر القياس الخاص بك

  def main():
    metrics.configure(MyMetricHandler())

    spec = WorkerSpec(...)
    agent = LocalElasticAgent(spec)
    agent.run()

معالج الأحداث
-----------------------------

يدعم TorchElastic تسجيل الأحداث (راجع :ref:`events-api`).
تعرف وحدة الأحداث واجهة برمجة التطبيقات التي تسمح لك بتسجيل الأحداث وتنفيذ معالج الأحداث المخصص. يستخدم معالج الأحداث لنشر الأحداث
التي يتم إنتاجها أثناء تنفيذ torchelastic إلى مصادر مختلفة، على سبيل المثال. سحابة أمازون ووتش.
بشكل افتراضي، يستخدم `torch.distributed.elastic.events.NullEventHandler` الذي يتجاهل
الأحداث. لتهيئة معالج أحداث مخصص، تحتاج إلى تنفيذ واجهة `torch.distributed.elastic.events.EventHandler` و`configure`
في مطلق التطبيق المخصص الخاص بك.

.. code-block:: python

  # my_launcher.py

  import torch.distributed.elastic.events as events

  class MyEventHandler(events.EventHandler):
      def record(self, event: events.Event):
          # معالجة الحدث

  def main():
    events.configure(MyEventHandler())

    spec = WorkerSpec(...)
    agent = LocalElasticAgent(spec)
    agent.run()