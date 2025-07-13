# 🏥 MedXpert - دليل التشغيل السريع

## 🚀 التشغيل على GitHub Codespaces (الطريقة الأسهل)

### الخطوات:
1. **إنشاء Codespace:**
   - اذهب إلى GitHub repository
   - اضغط "Code" → "Codespaces" → "Create codespace on main"

2. **تشغيل التطبيق:**
   ```bash
   python run.py
   ```
   أو:
   ```bash
   python app.py
   ```

3. **الحصول على الرابط:**
   - اذهب إلى تاب "PORTS" في VS Code
   - انسخ رابط المنفذ 5000
   - مثال: `https://fluffy-space-tribble-xyz123.github.dev`

## 🔗 روابط مهمة بعد التشغيل

| الصفحة | الرابط | الوصف |
|---------|---------|--------|
| الرئيسية | `/` | الصفحة الرئيسية |
| تحليل الدماغ | `/brain` | رفع صور NIFTI |
| تحليل الجلد | `/skin` | كشف سرطان الجلد |
| أشعة الصدر | `/chest` | كشف COVID-19 |
| كشف الكسور | `/fracture` | تحديد الكسور |
| ChatBot | `/chatbot` | المساعد الذكي |
| تسجيل الدخول | `/login` | الدخول للنظام |
| الإدارة | `/admin` | لوحة الإدارة |

## 🔐 بيانات دخول افتراضية

**المدير:**
- Username: `admin`
- Password: `admin123`

## ⚙️ أوامر مفيدة

```bash
# تثبيت المتطلبات
pip install -r requirements.txt

# تشغيل التطبيق
python app.py

# تشغيل مع إعادة التحميل التلقائي
FLASK_ENV=development python app.py

# فحص المنافذ المفتوحة
lsof -i :5000
```

## 📁 هيكل المشروع

```
medxpert/
├── app.py                 # التطبيق الرئيسي
├── run.py                 # تشغيل سريع
├── requirements.txt       # المتطلبات
├── startup.sh            # سكريبت التشغيل
├── models/               # النماذج الذكية
├── static/               # الملفات الثابتة
├── templates/            # قوالب HTML
├── .devcontainer/        # إعدادات Codespaces
└── .vscode/              # إعدادات VS Code
```

## 🔧 حل المشاكل الشائعة

### المشروع لا يعمل:
```bash
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### لا يمكن الوصول للموقع:
1. تحقق من تاب "PORTS" في VS Code
2. تأكد من أن المنفذ 5000 "Public"
3. انسخ الرابط الصحيح

### النماذج لا تعمل:
- تأكد من وجود ملفات النماذج في مجلد `models/`
- الملفات المطلوبة:
  - `best_metric_model.pth`
  - `best_model_skin.h5`
  - `best_model_chest.h5`
  - `best.pt`

## 📱 مشاركة التطبيق

بعد التشغيل، يمكنك مشاركة الرابط مع الآخرين:
```
https://[اسم-عشوائي]-5000.app.github.dev
```

⚠️ **ملاحظة:** الرابط يعمل فقط أثناء تشغيل Codespace

## 🆘 الدعم

في حالة وجود مشاكل:
1. تحقق من سجلات التطبيق في Terminal
2. راجع ملف `app.log`
3. تحقق من تاب "PROBLEMS" في VS Code
4. تأكد من تثبيت جميع المتطلبات بنجاح
