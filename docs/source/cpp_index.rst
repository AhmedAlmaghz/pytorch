C++
====
.. Note::
    إذا كنت تبحث عن وثائق PyTorch C++ API، انتقل مباشرة إلى `هنا <https://pytorch.org/cppdocs/>`__.

يوفر PyTorch العديد من الميزات للعمل مع C++، ومن الأفضل اختيار الأنسب منها بناءً على احتياجاتك. وعلى مستوى عالٍ، يتوفر الدعم التالي:

TorchScript C++ API
--------------------
يتيح `TorchScript <https://pytorch.org/docs/stable/jit.html>`__ إمكانية تسجيل نماذج PyTorch المحددة في Python، ثم تحميلها وتشغيلها في C++ عن طريق التقاط كود النموذج عبر التجميع أو تتبع تنفيذه. يمكنك معرفة المزيد في `برنامج تعليمي حول تحميل نموذج TorchScript في C++ <https://pytorch.org/tutorials/advanced/cpp_export.html>`__. وهذا يعني أنه يمكنك تحديد نماذجك في Python قدر الإمكان، ولكن بعد ذلك تصديرها عبر TorchScript للقيام بتنفيذ خالٍ من Python في بيئات الإنتاج أو الأنظمة المدمجة. وتستخدم TorchScript C++ API للتفاعل مع هذه النماذج ومحرك تنفيذ TorchScript، بما في ذلك:

* تحميل نماذج TorchScript المسجلة من Python
* إجراء تعديلات بسيطة على النموذج إذا لزم الأمر (مثل استخراج الوحدات الفرعية)
* إنشاء المدخلات وإجراء المعالجة المسبقة باستخدام C++ Tensor API

توسيع PyTorch و TorchScript باستخدام ملحقات C++
-------------------------------------------
يمكن تعزيز TorchScript باستخدام كود مخصص من خلال المشغلين المخصصين والصفوف المخصصة.
بمجرد تسجيلها في TorchScript، يمكن استدعاء هذه المشغلين والصفوف في كود TorchScript الذي يتم تشغيله من
Python أو من C++ كجزء من نموذج TorchScript المسجل. ويتناول البرنامج التعليمي "توسيع TorchScript باستخدام مشغلي C++ المخصصين" <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`__ عملية الربط بين TorchScript و OpenCV. بالإضافة إلى لف مكالمة دالة بمشغل مخصص، يمكن ربط صفوف C++ وهياكلها في TorchScript من خلال واجهة تشبه pybind11 والتي يتم شرحها في البرنامج التعليمي "توسيع TorchScript باستخدام صفوف C++ المخصصة" <https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html>`__.

Tensor و Autograd في C++
-------------------------
تتوفر معظم عمليات Tensor و Autograd في PyTorch Python API أيضًا في C++ API. وتشمل هذه ما يلي:

* طرق ``torch::Tensor`` مثل ``add`` / ``reshape`` / ``clone``. للحصول على القائمة الكاملة للطرق المتاحة، يرجى زيارة: https://pytorch.org/cppdocs/api/classat_1_1_tensor.html
* واجه برمجة تطبيقات فهرسة Tensor C++ التي تبدو وتتصرف بنفس طريقة واجه برمجة التطبيقات في Python. للحصول على تفاصيل حول استخدامها، يرجى زيارة: https://pytorch.org/cppdocs/notes/tensor_indexing.html
* واجهات برمجة تطبيقات Tensor Autograd وحزمة ``torch::autograd`` التي تعد بالغة الأهمية لبناء الشبكات العصبية الديناميكية في واجهة C++ الأمامية. لمزيد من التفاصيل، يرجى زيارة: https://pytorch.org/tutorials/advanced/cpp_autograd.html

إنشاء النماذج في C++
---------------
يتطلب سير عمل "الإنشاء في TorchScript، والاستنتاج في C++" أن يتم إنشاء النماذج في TorchScript.
ومع ذلك، قد تكون هناك حالات يتعين فيها إنشاء النموذج في C++ (على سبيل المثال، في سير العمل حيث يكون عنصر Python غير مرغوب فيه). ولخدمة مثل هذه الحالات الاستخدامية، نوفر القدرة الكاملة على إنشاء وتدريب نموذج الشبكة العصبية بالكامل في C++، مع مكونات مألوفة مثل ``torch::nn`` / ``torch::nn::functional`` / ``torch::optim`` التي تشبه واجه برمجة تطبيقات Python عن كثب.

* للحصول على نظرة عامة حول واجه برمجة تطبيقات PyTorch C++ لإنشاء النماذج والتدريب، يرجى زيارة: https://pytorch.org/cppdocs/frontend.html
* للحصول على برنامج تعليمي مفصل حول كيفية استخدام واجه برمجة التطبيقات، يرجى زيارة: https://pytorch.org/tutorials/advanced/cpp_frontend.html
* يمكن العثور على وثائق المكونات مثل ``torch::nn`` / ``torch::nn::functional`` / ``torch::optim`` في: https://pytorch.org/cppdocs/api/library_root.html


التغليف لـ C++
-----------
للحصول على إرشادات حول كيفية تثبيت والربط مع libtorch (المكتبة التي تحتوي على جميع واجهات برمجة تطبيقات C++ المذكورة أعلاه)، يرجى زيارة: https://pytorch.org/cppdocs/installing.html. لاحظ أنه في Linux، هناك نوعان من ثنائيات libtorch المقدمة: واحدة مجمعة مع GCC pre-cxx11 ABI والأخرى مع GCC cxx11 ABI، ويجب عليك إجراء الاختيار بناءً على ABI GCC الذي يستخدمه نظامك.