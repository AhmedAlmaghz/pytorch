بدء الاستخدام
===========

لتشغيل مهمة **متسامحة مع الأخطاء** ، قم بتشغيل ما يلي على جميع العقد.

.. code-block:: bash

    torchrun
       --nnodes=NUM_NODES
       --nproc-per-node=TRAINERS_PER_NODE
       --max-restarts=NUM_ALLOWED_FAILURES
       --rdzv-id=JOB_ID
       --rdzv-backend=c10d
       --rdzv-endpoint=HOST_NODE_ADDR
       YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)


لتشغيل مهمة **مرنة** ، قم بتشغيل ما يلي على الأقل ``MIN_SIZE`` عقد
وعلى الأكثر ``MAX_SIZE`` عقد.

.. code-block:: bash

    torchrun
        --nnodes=MIN_SIZE:MAX_SIZE
        --nproc-per-node=TRAINERS_PER_NODE
        --max-restarts=NUM_ALLOWED_FAILURES_OR_MEMBERSHIP_CHANGES
        --rdzv-id=JOB_ID
        --rdzv-backend=c10d
        --rdzv-endpoint=HOST_NODE_ADDR
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

.. note::
   تعتبر TorchElastic الأخطاء على أنها تغييرات في العضوية. عندما يفشل عقدة ،
   يتم التعامل مع هذا على أنه حدث "الخفض". عندما يستبدل المجدول العقدة الفاشلة ، فهو حدث "التصعيد". وبالتالي ، لكل من المهام المتسامحة مع الأخطاء
   والمرنة ، يتم استخدام ``--max-restarts`` للتحكم في العدد الإجمالي
   إعادة التشغيل قبل الاستسلام ، بغض النظر عما إذا كان إعادة التشغيل ناتجًا
   بسبب فشل أو حدث تغيير الحجم.

``HOST_NODE_ADDR`` ، على الشكل <host> [: <port>] (على سبيل المثال node1.example.com:29400) ،
يحدد العقدة والمنفذ الذي يجب أن يتم عليه إنشاء وتشغيل C10d
خلفية اللقاء. يمكن أن تكون أي عقدة في مجموعة العقد التدريبية الخاصة بك ، ولكن
من الناحية المثالية ، يجب عليك اختيار عقدة ذات نطاق ترددي عالي.

.. note::
   إذا لم يتم تحديد رقم المنفذ ، فإن ``HOST_NODE_ADDR`` الافتراضي هو 29400.

.. note::
   يمكن تمرير خيار ``--standalone`` لبدء مهمة عقدة واحدة مع
   خادم واجهة برمجة التطبيقات اللقاء. لا يلزم تمرير ``--rdzv-id`` ،
   ``--rdzv-endpoint`` ، و ``--rdzv-backend`` عند استخدام
   يتم استخدام خيار "مستقل".

.. note::
   تعرف على المزيد حول كتابة نص البرنامج النصي الموزع
   `هنا <train_script.html>`_.

إذا لم يكن ``torchrun`` متوافقًا مع متطلباتك ، فيمكنك استخدام واجهات برمجة التطبيقات الخاصة بنا مباشرةً
لمزيد من التخصيص القوي. ابدأ بإلقاء نظرة على
واجهة برمجة تطبيقات `elastic agent <agent.html>`_ .