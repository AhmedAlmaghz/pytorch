.. _python-language-reference:

مرجع لغة بايثون
هذا جدول يوضح مطابقة 1:1 للميزات المدرجة في https://docs.python.org/3/reference/ ودعمها في TorchScript. التصنيفات هي كما يلي:

.. list-table::
   :header-rows: 1
بالتأكيد! فيما يلي النص المترجم إلى اللغة العربية مع الحفاظ على تنسيق ReStructuredText:

* - قسم
     - الحالة
     - ملاحظة
   * - `1. مقدمة <https://docs.python.org/3/reference/introduction.html>`_
     - غير ذي صلة
     -
   * - `1.1. التنفيذ البديل <https://docs.python.org/3/reference/introduction.html#alternate-implementations>`_
     - غير ذي صلة
     -
   * - `1.2. التدوين <https://docs.python.org/3/reference/introduction.html#notation>`_
     - غير ذي صلة
     -
   * - `2. التحليل القاموسي <https://docs.python.org/3/reference/lexical_analysis.html#>`_
     - غير ذي صلة
     -
   * - `2.1. هيكل السطر <https://docs.python.org/3/reference/lexical_analysis.html#line-structure>`_
     - غير ذي صلة
     -
   * - `2.1.1. الأسطر المنطقية <https://docs.python.org/3/reference/lexical_analysis.html#logical-lines>`_
     - غير ذي صلة
     -
   * - `2.1.2. الأسطر الفعلية <https://docs.python.org/3/reference/lexical_analysis.html#physical-lines>`_
     - مدعوم
     -
   * - `2.1.3. التعليقات <https://docs.python.org/3/reference/lexical_analysis.html#comments>`_
     - مدعوم
     -
   * - `2.1.4. إعلانات الترميز <https://docs.python.org/3/reference/lexical_analysis.html#encoding-declarations>`_
     - غير مدعوم
     - TorchScript لا يدعم unicode بشكل صريح
   * - `2.1.5. الانضمام الصريح للأسطر <https://docs.python.org/3/reference/lexical_analysis.html#explicit-line-joining>`_
     - مدعوم
     -
   * - `2.1.6. الانضمام الضمني للأسطر <https://docs.python.org/3/reference/lexical_analysis.html#implicit-line-joining>`_
     - مدعوم
     -
   * - `2.1.7. الأسطر الفارغة <https://docs.python.org/3/reference/lexical_analysis.html#blank-lines>`_
     - مدعوم
     -
   * - `2.1.8. الإزاحة <https://docs.python.org/3/reference/lexical_analysis.html#indentation>`_
     - مدعوم
     -
   * - `2.1.9. المسافات البيضاء بين الرموز <https://docs.python.org/3/reference/lexical_analysis.html#whitespace-between-tokens>`_
     - غير ذي صلة
     -
   * - `2.2. الرموز الأخرى <https://docs.python.org/3/reference/lexical_analysis.html#other-tokens>`_
     - غير ذي صلة
     -
   * - `2.3. المعرفات والكلمات المحجوزة <https://docs.python.org/3/reference/lexical_analysis.html#identifiers>`_
     - مدعوم
     -
   * - `2.3.1. الكلمات المحجوزة <https://docs.python.org/3/reference/lexical_analysis.html#keywords>`_
     - مدعوم
     -
   * - `2.3.2. الفئات المحجوزة من المعرفات <https://docs.python.org/3/reference/lexical_analysis.html#reserved-classes-of-identifiers>`_
     - مدعوم
     -
   * - `2.4. الحرفي <https://docs.python.org/3/reference/lexical_analysis.html#literals>`_
     - غير ذي صلة
     -
   * - `2.4.1. الحرفي السلسلة والبايتات <https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals>`_
     - مدعوم
     -
   * - `2.4.2. تجميع حروف السلسلة <https://docs.python.org/3/reference/lexical_analysis.html#string-literal-concatenation>`_
     - مدعوم
     -
   * - `2.4.3. الحرفي السلسلة المنسقة <https://docs.python.org/3/reference/lexical_analysis.html#formatted-string-literals>`_
     - مدعوم جزئيا
     -
   * - `2.4.4. الحرفي الرقمي <https://docs.python.org/3/reference/lexical_analysis.html#numeric-literals>`_
     - مدعوم
     -
   * - `2.4.5. الحرفي الرقمي الصحيح <https://docs.python.org/3/reference/lexical_analysis.html#integer-literals>`_
     - مدعوم
     -
   * - `2.4.6. الحرفي الرقمي العشري <https://docs.python.org/3/reference/lexical_analysis.html#floating-point-literals>`_
     - مدعوم
     -
   * - `2.4.7. الحرفي التخيلي <https://docs.python.org/3/reference/lexical_analysis.html#imaginary-literals>`_
     - غير مدعوم
     -
   * - `2.5. المشغلون <https://docs.python.org/3/reference/lexical_analysis.html#operators>`_
     - مدعوم جزئيا
     - غير مدعوم: ``<<``، ``>>``، ``:=``
   * - `2.6. الفواصل <https://docs.python.org/3/reference/lexical_analysis.html#delimiters>`_
     - مدعوم جزئيا
     - غير مدعوم: ``**=``، ``<<=``، ``>>=``، ``%=``، ``^=``، ``@=``، ``&=``، ``//=``، عامل التشغيل ``%`` لبعض الأنواع (على سبيل المثال ``str``)
   * - `3. نموذج البيانات <https://docs.python.org/3/reference/datamodel.html#>`_
     - غير ذي صلة
     -
   * - `3.1. الكائنات والقيم والأنواع <https://docs.python.org/3/reference/datamodel.html#objects-values-and-types>`_
     - غير ذي صلة
     -
   * - `3.2. التسلسل الهرمي القياسي للأنواع <https://docs.python.org/3/reference/datamodel.html#the-standard-type-hierarchy>`_
     - مدعوم جزئيا
     - غير مدعوم: NotImplemented، Ellipsis، numbers.Complex، bytes، مصفوفات البايت، المجموعات، المجموعات المجمدة، المولدات، الروتينات، المولدات غير المتزامنة، المولدات غير المتزامنة، الوحدات النمطية، كائنات الإدخال/الإخراج، الكائنات الداخلية، كائنات الشرائح (على الرغم من دعم الشرائح)، classmethod
   * - `3.3. أسماء الطرق الخاصة <https://docs.python.org/3/reference/datamodel.html#special-method-names>`_
     - مدعوم
     -
   * - `3.3.1. التخصيص الأساسي <https://docs.python.org/3/reference/datamodel.html#basic-customization>`_
     - مدعوم جزئيا
     - غير مدعوم: ``__new__``، ``__del__``، ``__bytes__``، ``__format__``، ``__hash__``
   * - `3.3.2. تخصيص الوصول إلى المعرف <https://docs.python.org/3/reference/datamodel.html#customizing-attribute-access>`_
     - غير مدعوم
     -
   * - `3.3.2.1. تخصيص الوصول إلى معرف الوحدة النمطية <https://docs.python.org/3/reference/datamodel.html#customizing-module-attribute-access>`_
     - غير مدعوم
     -
   * - `3.3.2.2. تنفيذ الواصفات <https://docs.python.org/3/reference/datamodel.html#implementing-descriptors>`_
     - غير مدعوم
     -
   * - `3.3.2.3. استدعاء الواصفات <https://docs.python.org/3/reference/datamodel.html#invoking-descriptors>`_
     - غير مدعوم
     -
   * - `3.3.2.4. __slots__ <https://docs.python.org/3/reference/datamodel.html#slots>`_
     - غير مدعوم
     -
   * - `3.3.2.4.1. ملاحظات حول استخدام __slots__ <https://docs.python.org/3/reference/datamodel.html#notes-on-using-slots>`_
     - غير مدعوم
     -
   * - `3.3.3. تخصيص إنشاء الفئات <https://docs.python.org/3/reference/datamodel.html#customizing-class-creation>`_
     - غير مدعوم
     -
   * - `3.3.3.1. الفئات الفوقية <https://docs.python.org/3/reference/datamodel.html#metaclasses>`_
     - غير مدعوم
     -
   * - `3.3.3.2. حل إدخالات MRO <https://docs.python.org/3/reference/datamodel.html#resolving-mro-entries>`_
     - غير مدعوم
     - ``super()`` غير مدعوم
   * - `3.3.3.3. تحديد الفئة الفوقية المناسبة <https://docs.python.org/3/reference/datamodel.html#determining-the-appropriate-metaclass>`_
     - غير ذي صلة
     -
   * - `3.3.3.4. إعداد مساحة اسم الفئة <https://docs.python.org/3/reference/datamodel.html#preparing-the-class-namespace>`_
     - غير ذي صلة
     -
   * - `3.3.3.5. تنفيذ جسم الفئة <https://docs.python.org/3/reference/datamodel.html#executing-the-class-body>`_
     - غير ذي صلة
     -
   * - `3.3.3.6. إنشاء كائن الفئة <https://docs.python.org/3/reference/datamodel.html#creating-the-class-object>`_
     - غير ذي صلة
     -
   * - `3.3.3.7. استخدامات الفئات الفوقية <https://docs.python.org/3/reference/datamodel.html#uses-for-metaclasses>`_
     - غير ذي صلة
     -
   * - `3.3.4. تخصيص التحقق من المثيل والصنف الفرعي <https://docs.python.org/3/reference/datamodel.html#customizing-instance-and-subclass-checks>`_
     - غير مدعوم
     -
   * - `3.3.5. محاكاة الأنواع العامة <https://docs.python.org/3/reference/datamodel.html#emulating-generic-types>`_
     - غير مدعوم
     -
   * - `3.3.6. محاكاة الكائنات القابلة للاستدعاء <https://docs.python.org/3/reference/datamodel.html#emulating-callable-objects>`_
     - مدعوم
     -
   * - `3.3.7. محاكاة أنواع الحاويات <https://docs.python.org/3/reference/datamodel.html#emulating-container-types>`_
     - مدعوم جزئيا
     - بعض الطرق السحرية غير مدعومة (على سبيل المثال ``__iter__``)
   * - `3.3.8. محاكاة الأنواع الرقمية <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
     - مدعوم جزئيا
     - الطرق السحرية مع معاملات التشغيل المبدلة غير مدعومة (``__r*__``)
   * - `3.3.9. مدراء سياق عبارة with <https://docs.python.org/3/reference/datamodel.html#with-statement-context-managers>`_
     - غير مدعوم
     -
   * - `3.3.10. البحث عن الطريقة الخاصة <https://docs.python.org/3/reference/datamodel.html#special-method-lookup>`_
     - غير ذي صلة
     -
   * - `3.4. الروتينات <https://docs.python.org/3/reference/datamodel.html#coroutines>`_
     - غير مدعوم
     -
   * - `3.4.1. كائنات قابلة للانتظار <https://docs.python.org/3/reference/datamodel.html#awaitable-objects>`_
     - غير مدعوم
     -
   * - `3.4.2. كائنات الروتين <https://docs.python.org/3/reference/datamodel.html#coroutine-objects>`_
     - غير مدعوم
     -
   * - `3.4.3. المؤشرات المتكررة غير المتزامنة <https://docs.python.org/3/reference/datamodel.html#asynchronous-iterators>`_
     - غير مدعوم
     -
   * - `3.4.4. مدراء السياق غير المتزامن <https://docs.python.org/3/reference/datamodel.html#asynchronous-context-managers>`_
     - غير مدعوم
     -
   * - `4. نموذج التنفيذ <https://docs.python.org/3/reference/executionmodel.html#>`_
     - غير ذي صلة
     -
   * - `4.1. هيكل البرنامج <https://docs.python.org/3/reference/executionmodel.html#structure-of-a-program>`_
     - غير ذي صلة
     -
   * - `4.2. التسمية والربط <https://docs.python.org/3/reference/executionmodel.html#naming-and-binding>`_
     - غير ذي صلة
     - ترتبط الأسماء في وقت الترجمة في TorchScript
   * - `4.2.1. ربط الأسماء <https://docs.python.org/3/reference/executionmodel.html#binding-of-names>`_
     - غير ذي صلة
     - راجع قسم تعليمات ``global`` و ``nonlocal``
   * - `4.2.2. حل الأسماء <https://docs.python.org/3/reference/executionmodel.html#resolution-of-names>`_
     - غير ذي صلة
     - راجع قسم تعليمات ``global`` و ``nonlocal``
   * - `4.2.3. المضمنات والتنفيذ المقيد <https://docs.python.org/3/reference/executionmodel.html#builtins-and-restricted-execution>`_
     - غير ذي صلة
     -
   * - `4.2.4. التفاعل مع الميزات الديناميكية <https://docs.python.org/3/reference/executionmodel.html#interaction-with-dynamic-features>`_
     - غير مدعوم
     - لا يمكن التقاط قيم Python
   * - `4.3. الاستثناءات <https://docs.python.org/3/reference/executionmodel.html#exceptions>`_
     - مدعوم جزئيا
     - راجع قسم تعليمات ``try`` و ``raise``
   * - `5. نظام الاستيراد <https://docs.python.org/3/reference/import.html>`_
     - غير ذي صلة
     -
   * - `6. التعبيرات <https://docs.python.org/3/reference/expressions.html#>`_
     - غير ذي صلة
     - راجع قسم التعبيرات
   * - `6.1. التحويلات الحسابية <https://docs.python.org/3/reference/expressions.html#arithmetic-conversions>`_
     - مدعوم
     -
   * - `6.2. الذرات <https://docs.python.org/3/reference/expressions.html#atoms>`_
     - غير ذي صلة
     -
   * - `6.2.1. المعرفات (الأسماء) <https://docs.python.org/3/reference/expressions.html#atom-identifiers>`_
     - مدعوم
     -
   * - `6.2.2. الحرفي <https://docs.python.org/3/reference/expressions.html#literals>`_
     - مدعوم جزئيا
     - غير مدعوم: ``bytesliteral``، ``imagnumber``
   * - `6.2.3. الأشكال الموضوعة بين قوسين <https://docs.python.org/3/reference/expressions.html#parenthesized-forms>`_
     - مدعوم
     -
   * - `6.2.4. العروض الخاصة بالقوائم والمجموعات والقواميس <https://docs.python.org/3/reference/expressions.html#displays-for-lists-sets-and-dictionaries>`_
     - مدعوم جزئيا
     - غير مدعوم: ifs الفهم، المؤشرات المتكررة غير المتزامنة
   * - `6.2.5. عروض القوائم <https://docs.python.org/3/reference/expressions.html#list-displays>`_
     - مدعوم
     -
   * - `6.2.6. عروض المجموعات <https://docs.python.org/
بالتأكيد، إليك النص المترجم إلى اللغة العربية مع الحفاظ على تنسيق ReStructuredText:

* - `6.2.9. تعابير العائد <https://docs.python.org/3/reference/expressions.html#yield-expressions>`_
     - غير مدعوم
     -
   * - `6.2.9.1. أساليب مولد المؤشر <https://docs.python.org/3/reference/expressions.html#generator-iterator-methods>`_
     - غير مدعوم
     -
   * - `6.2.9.2. أمثلة <https://docs.python.org/3/reference/expressions.html#examples>`_
     - غير مدعوم
     -
   * - `6.2.9.3. وظائف المولدات غير المتزامنة <https://docs.python.org/3/reference/expressions.html#asynchronous-generator-functions>`_
     - غير مدعوم
     -
   * - `6.2.9.4. أساليب مولد المؤشر غير المتزامن <https://docs.python.org/3/reference/expressions.html#asynchronous-generator-iterator-methods>`_
     - غير مدعوم
     -
   * - `6.3. الأساسيات <https://docs.python.org/3/reference/expressions.html#primaries>`_
     - مدعوم
     -
   * - `6.3.1. مراجع السمات <https://docs.python.org/3/reference/expressions.html#attribute-references>`_
     - مدعوم
     -
   * - `6.3.2. الاشتراكات <https://docs.python.org/3/reference/expressions.html#subscriptions>`_
     - مدعوم
     -
   * - `6.3.3. الشرائح <https://docs.python.org/3/reference/expressions.html#slicings>`_
     - مدعوم جزئياً
     - شرائح المصفوفات مع الخطوة غير مدعومة
   * - `6.3.4. المكالمات <https://docs.python.org/3/reference/expressions.html#calls>`_
     - مدعوم جزئياً
     - فك حزم الحجج وحزم الكلمات الرئيسية غير مدعوم
   * - `6.4. تعبير الانتظار <https://docs.python.org/3/reference/expressions.html#await-expression>`_
     - غير مدعوم
     -
   * - `6.5. عامل القوة <https://docs.python.org/3/reference/expressions.html#the-power-operator>`_
     - مدعوم
     -
   * - `6.6. العمليات الحسابية والبتية الأحادية <https://docs.python.org/3/reference/expressions.html#unary-arithmetic-and-bitwise-operations>`_
     - مدعوم جزئياً
     - بعض عوامل البت غير منفذة للأنواع الأولية (على سبيل المثال، "~x" حيث "x" هو عدد صحيح غير مدعوم حالياً)
   * - `6.7. العمليات الحسابية الثنائية <https://docs.python.org/3/reference/expressions.html#binary-arithmetic-operations>`_
     - مدعوم جزئياً
     - راجع قسم الفواصل
   * - `6.8. عمليات التحول <https://docs.python.org/3/reference/expressions.html#shifting-operations>`_
     - غير مدعوم
     -
   * - `6.9. العمليات الثنائية للبت <https://docs.python.org/3/reference/expressions.html#binary-bitwise-operations>`_
     - مدعوم
     -
   * - `6.10. المقارنات <https://docs.python.org/3/reference/expressions.html#comparisons>`_
     - مدعوم
     -
   * - `6.10.1. مقارنات القيمة <https://docs.python.org/3/reference/expressions.html#value-comparisons>`_
     - مدعوم جزئياً
     - فحوصات المساواة للقاموس غير مدعومة حالياً
   * - `6.10.2. عمليات اختبار العضوية <https://docs.python.org/3/reference/expressions.html#membership-test-operations>`_
     - مدعوم جزئياً
     - غير مدعوم لفئات TorchScript
   * - `6.10.3. مقارنات الهوية <https://docs.python.org/3/reference/expressions.html#is-not>`_
     - مدعوم
     -
   * - `6.11. العمليات المنطقية <https://docs.python.org/3/reference/expressions.html#boolean-operations>`_
     - مدعوم
     -
   * - `6.12. التعابير الشرطية <https://docs.python.org/3/reference/expressions.html#conditional-expressions>`_
     - مدعوم
     -
   * - `6.13. لامبداس <https://docs.python.org/3/reference/expressions.html#lambda>`_
     - غير مدعوم
     -
   * - `6.14. قوائم التعبير <https://docs.python.org/3/reference/expressions.html#expression-lists>`_
     - مدعوم جزئياً
     - فك حزم المصفوفات غير مدعوم
   * - `6.15. ترتيب التقييم <https://docs.python.org/3/reference/expressions.html#evaluation-order>`_
     - مدعوم
     -
   * - `6.16. أسبقية المشغل <https://docs.python.org/3/reference/expressions.html#operator-precedence>`_
     - مدعوم
     -
   * - `7. العبارات البسيطة <https://docs.python.org/3/reference/simple_stmts.html#>`_
     - مدعوم
     -
   * - `7.1. عبارات التعبير <https://docs.python.org/3/reference/simple_stmts.html#expression-statements>`_
     - مدعوم
     -
   * - `7.2. عبارات التعيين <https://docs.python.org/3/reference/simple_stmts.html#assignment-statements>`_
     - مدعوم
     -
   * - `7.2.1. عبارات التعيين المعززة <https://docs.python.org/3/reference/simple_stmts.html#augmented-assignment-statements>`_
     - مدعوم جزئياً
     - راجع قسم الفواصل
   * - `7.2.2. عبارات التعيين المشروح <https://docs.python.org/3/reference/simple_stmts.html#annotated-assignment-statements>`_
     - مدعوم
     -
   * - `7.3. عبارة التأكيد <https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement>`_
     - مدعوم جزئياً
     - لا يمكن تخصيص رسالة الاستثناء
   * - `7.4. عبارة المرور <https://docs.python.org/3/reference/simple_stmts.html#the-pass-statement>`_
     - مدعوم
     -
   * - `7.5. عبارة الحذف <https://docs.python.org/3/reference/simple_stmts.html#the-del-statement>`_
     - غير مدعوم
     -
   * - `7.6. عبارة الإرجاع <https://docs.python.org/3/reference/simple_stmts.html#the-return-statement>`_
     - مدعوم
     - بعض الميزات الأخرى للإرجاع (مثل السلوك مع try..finally) غير مدعومة
   * - `7.7. عبارة العائد <https://docs.python.org/3/reference/simple_stmts.html#the-yield-statement>`_
     - غير مدعوم
     -
   * - `7.8. عبارة إثارة الاستثناء <https://docs.python.org/3/reference/simple_stmts.html#the-raise-statement>`_
     - مدعوم جزئياً
     - لا يمكن تخصيص رسالة الاستثناء
   * - `7.9. عبارة كسر <https://docs.python.org/3/reference/simple_stmts.html#the-break-statement>`_
     - مدعوم
     - بعض الميزات الأخرى للإرجاع (مثل السلوك مع try..finally) غير مدعومة
   * - `7.10. عبارة الاستمرار <https://docs.python.org/3/reference/simple_stmts.html#the-continue-statement>`_
     - مدعوم
     - بعض الميزات الأخرى للإرجاع (مثل السلوك مع try..finally) غير مدعومة
   * - `7.11. عبارة الاستيراد <https://docs.python.org/3/reference/simple_stmts.html#the-import-statement>`_
     - غير مدعوم
     -
   * - `7.11.1. عبارات المستقبل <https://docs.python.org/3/reference/simple_stmts.html#future-statements>`_
     - غير مدعوم
     -
   * - `7.12. عبارة العمومية <https://docs.python.org/3/reference/simple_stmts.html#the-global-statement>`_
     - غير مدعوم
     -
   * - `7.13. عبارة المحلية غير المدعومة <https://docs.python.org/3/reference/simple_stmts.html#the-nonlocal-statement>`_
     - غير مدعوم
     -
   * - `8. العبارات المركبة <https://docs.python.org/3/reference/compound_stmts.html#>`_
     - غير ذي صلة
     -
   * - `8.1. عبارة إذا <https://docs.python.org/3/reference/compound_stmts.html#the-if-statement>`_
     - مدعوم
     -
   * - `8.2. عبارة أثناء <https://docs.python.org/3/reference/compound_stmts.html#the-while-statement>`_
     - مدعوم جزئياً
     - while..else غير مدعوم
   * - `8.3. عبارة بالنسبة <https://docs.python.org/3/reference/compound_stmts.html#the-for-statement>`_
     - مدعوم جزئياً
     - for..else غير مدعوم
   * - `8.4. جرب عبارة <https://docs.python.org/3/reference/compound_stmts.html#the-try-statement>`_
     - غير مدعوم
     -
   * - `8.5. عبارة مع <https://docs.python.org/3/reference/compound_stmts.html#the-with-statement>`_
     - مدعوم جزئياً
     - "__exit__" يتم استدعاؤه دائماً مع "exc_type" و "exc_value" و "traceback" يتم تعيينها إلى None، حتى إذا تم إثارة استثناء، وتتم تجاهل قيمة الإرجاع لـ "__exit__".
   * - `8.6. تعريفات الدالة <https://docs.python.org/3/reference/compound_stmts.html#function-definitions>`_
     - غير مدعوم
     -
   * - `8.7. تعريفات الفئة <https://docs.python.org/3/reference/compound_stmts.html#class-definitions>`_
     - غير مدعوم
     -
   * - `8.8. الروتينات الفرعية <https://docs.python.org/3/reference/compound_stmts.html#coroutines>`_
     - غير مدعوم
     -
   * - `8.8.1. تعريف روتين فرعي للدالة <https://docs.python.org/3/reference/compound_stmts.html#coroutine-function-definition>`_
     - غير مدعوم
     -
   * - `8.8.2. عبارة for غير المتزامنة <https://docs.python.org/3/reference/compound_stmts.html#the-async-for-statement>`_
     - غير مدعوم
     -
   * - `8.8.3. عبارة with غير المتزامنة <https://docs.python.org/3/reference/compound_stmts.html#the-async-with-statement>`_
     - غير مدعوم
     -
   * - `9. المكونات على مستوى أعلى <https://docs.python.org/3/reference/toplevel_components.html#>`_
     - غير ذي صلة
     -
   * - `9.1. برامج Python الكاملة <https://docs.python.org/3/reference/toplevel_components.html#complete-python-programs>`_
     - غير ذي صلة
     -
   * - `9.2. إدخال الملف <https://docs.python.org/3/reference/toplevel_components.html#file-input>`_
     - غير ذي صلة
     -
   * - `9.3. الإدخال التفاعلي <https://docs.python.org/3/reference/toplevel_components.html#interactive-input>`_
     - غير ذي صلة
     -
   * - `9.4. إدخال التعبير <https://docs.python.org/3/reference/toplevel_components.html#expression-input>`_
     - غير ذي صلة
     -