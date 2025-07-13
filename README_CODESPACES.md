# 🏥 MedXpert - GitHub Codespaces Setup

## 🚀 Quick Start

### تشغيل المشروع على GitHub Codespaces

1. **إنشاء Codespace جديد:**
   - اذهب إلى repository الخاص بك على GitHub
   - اضغط على زر "Code" الأخضر
   - اختر "Codespaces" 
   - اضغط "Create codespace on main"

2. **تشغيل التطبيق:**
   ```bash
   chmod +x startup.sh
   ./startup.sh
   ```

   أو يمكنك تشغيله مباشرة:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```

3. **الوصول للموقع:**
   - سيظهر إشعار في VS Code بأن التطبيق يعمل على المنفذ 5000
   - اضغط "Open in Browser" للوصول للموقع
   - أو اذهب إلى تاب "PORTS" في VS Code واضغط على رابط المنفذ 5000

## 🔗 الحصول على رابط الموقع

بعد تشغيل التطبيق، ستحصل على رابط مثل:
```
https://[random-name]-5000.app.github.dev
```

يمكنك مشاركة هذا الرابط مع الآخرين للوصول للموقع.

## ⚙️ المميزات المتاحة

- **تحليل صور الدماغ**: رفع ملفات NIFTI لكشف الأورام
- **تحليل صور الجلد**: كشف سرطان الجلد من الصور
- **تحليل أشعة الصدر**: كشف COVID-19 والتهاب الرئة
- **كشف الكسور**: تحديد الكسور في صور الأشعة
- **ChatBot طبي**: نظام ذكي للاستشارات الطبية

## 🔐 بيانات الدخول الافتراضية

**Admin:**
- Username: `admin`
- Password: `admin123`

## 📝 ملاحظات مهمة

1. **الملفات الكبيرة**: النماذج الذكية قد تحتاج وقت للتحميل في المرة الأولى
2. **المساحة**: تأكد أن Codespace لديه مساحة كافية للنماذج
3. **الأداء**: قد يكون أبطأ من التشغيل المحلي

## 🔧 إعدادات إضافية

### متغيرات البيئة (اختيارية)
```bash
export GOOGLE_API_KEY="your_gemini_api_key"  # للChatBot
export FLASK_ENV="production"  # للإنتاج
```

### إضافة نماذج ذكية
1. ارفع النماذج إلى مجلد `models/`
2. تأكد من الأسماء الصحيحة:
   - `best_metric_model.pth` (Brain)
   - `best_model_skin.h5` (Skin)
   - `best_model_chest.h5` (Chest)
   - `best.pt` (Fracture)

## 🆘 حل المشاكل

### مشكلة في تشغيل المشروع:
```bash
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### مشكلة في الوصول للموقع:
1. تأكد من أن المنفذ 5000 مفتوح
2. اذهب إلى تاب "PORTS" في VS Code
3. تأكد من أن الرؤية مضبوطة على "Public"

## 📞 الدعم

في حالة وجود مشاكل، تحقق من:
- سجلات التطبيق في terminal
- ملف `app.log` للأخطاء
- تاب "PROBLEMS" في VS Code
