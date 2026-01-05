# 📦 دليل التثبيت الكامل

## 🖥️ متطلبات النظام

### الحد الأدنى
- **نظام التشغيل**: Windows 10/11, Linux, macOS
- **Python**: 3.8 أو أحدث
- **الذاكرة**: 4 GB RAM
- **المساحة**: 2 GB قرص صلب
- **معالج**: Intel i3 أو ما يعادله

### الموصى به
- **الذاكرة**: 8 GB RAM أو أكثر
- **معالج**: Intel i5 أو أفضل
- **GPU**: NVIDIA GPU مع CUDA (اختياري للتدريب الأسرع)

---

## 📥 التثبيت خطوة بخطوة

### الطريقة 1: التثبيت التلقائي (Windows) ⭐ موصى به

```batch
# 1. نزّل المشروع
# 2. افتح مجلد المشروع
# 3. شغّل run.bat
# 4. اختر "تثبيت المكتبات المطلوبة" (خيار 1)
run.bat
```

### الطريقة 2: التثبيت اليدوي

#### Windows

```powershell
# 1. فتح PowerShell أو Command Prompt
cd C:\Users\YourName\Desktop\Lung_cancer_detector

# 2. التحقق من Python
python --version
# يجب أن يكون 3.8 أو أحدث

# 3. إنشاء بيئة افتراضية (موصى به)
python -m venv venv

# 4. تفعيل البيئة الافتراضية
venv\Scripts\activate

# 5. تحديث pip
python -m pip install --upgrade pip

# 6. تثبيت المكتبات
pip install -r requirements.txt

# 7. (اختياري) تثبيت أدوات التطوير
pip install -r requirements-dev.txt
```

#### Linux / macOS

```bash
# 1. فتح Terminal
cd ~/Desktop/Lung_cancer_detector

# 2. التحقق من Python
python3 --version
# يجب أن يكون 3.8 أو أحدث

# 3. إنشاء بيئة افتراضية
python3 -m venv venv

# 4. تفعيل البيئة الافتراضية
source venv/bin/activate

# 5. تحديث pip
python -m pip install --upgrade pip

# 6. تثبيت المكتبات
pip install -r requirements.txt

# 7. (اختياري) تثبيت أدوات التطوير
pip install -r requirements-dev.txt
```

---

## 🎮 تثبيت PyTorch مع GPU (اختياري)

إذا كان لديك GPU من NVIDIA وتريد تسريع التدريب:

### 1. تحقق من CUDA

```bash
# Windows
nvidia-smi

# Linux
nvidia-smi
```

### 2. تثبيت PyTorch مع CUDA

```bash
# CUDA 11.8 (الأكثر استقراراً)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# أو CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# للتحقق من التثبيت
python -c "import torch; print(torch.cuda.is_available())"
# يجب أن يطبع: True
```

### 3. بدون GPU (CPU فقط)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## ✅ التحقق من التثبيت

### 1. اختبار سريع

```bash
# تفعيل البيئة الافتراضية أولاً
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# تشغيل Python
python
```

```python
# في Python، جرّب:
import torch
import streamlit
import numpy
import sklearn
import PIL
import matplotlib
import seaborn

print("✅ جميع المكتبات مثبتة بنجاح!")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA متاح: {torch.cuda.is_available()}")
```

### 2. اختبار شامل

```bash
# تشغيل ملف الإعدادات
python config.py

# يجب أن يعرض معلومات المشروع والإعدادات
```

---

## 🔧 حل مشاكل التثبيت الشائعة

### ❌ مشكلة: Python غير معروف

**الحل:**
1. تأكد من تثبيت Python
2. أضف Python إلى PATH
3. أعد تشغيل الـ Terminal/CMD

```bash
# Windows: أضف إلى PATH
C:\Users\YourName\AppData\Local\Programs\Python\Python311
C:\Users\YourName\AppData\Local\Programs\Python\Python311\Scripts
```

### ❌ مشكلة: pip غير معروف

**الحل:**
```bash
# استخدم
python -m pip install package_name

# بدلاً من
pip install package_name
```

### ❌ مشكلة: خطأ في تثبيت torch

**الحل:**
```bash
# 1. احذف الإصدار الحالي
pip uninstall torch torchvision

# 2. ثبّت من الموقع الرسمي
# CPU فقط
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# أو مع CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### ❌ مشكلة: خطأ في الصلاحيات (Linux/Mac)

**الحل:**
```bash
# استخدم --user
pip install --user -r requirements.txt

# أو استخدم sudo (غير موصى به)
sudo pip install -r requirements.txt
```

### ❌ مشكلة: مساحة غير كافية

**الحل:**
- احذف ملفات مؤقتة: `pip cache purge`
- استخدم `--no-cache-dir`: `pip install --no-cache-dir package`

### ❌ مشكلة: اتصال بطيء

**الحل:**
```bash
# استخدم مرآة صينية
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# أو مرآة أخرى
pip install -r requirements.txt -i https://pypi.org/simple
```

---

## 🌐 التثبيت في بيئات مختلفة

### Google Colab

```python
# في خلية جديدة
!git clone https://github.com/yourusername/lung_cancer_detector
%cd lung_cancer_detector
!pip install -r requirements.txt
```

### Anaconda

```bash
# إنشاء بيئة جديدة
conda create -n lung_cancer python=3.9

# تفعيل البيئة
conda activate lung_cancer

# تثبيت PyTorch
conda install pytorch torchvision -c pytorch

# تثبيت البقية
pip install -r requirements.txt
```

### Docker

```dockerfile
# قريباً...
# سيتم إضافة Dockerfile
```

---

## 📦 قائمة المكتبات المطلوبة

### المكتبات الأساسية

| المكتبة | الإصدار | الاستخدام |
|---------|---------|-----------|
| torch | 2.0.1 | التعلم العميق |
| torchvision | 0.15.2 | معالجة الصور |
| Pillow | 10.0.0 | التعامل مع الصور |
| numpy | 1.24.3 | العمليات الرياضية |
| streamlit | 1.28.1 | واجهة المستخدم |
| scikit-learn | 1.3.2 | التعلم الآلي |
| scipy | 1.11.4 | عمليات علمية |
| matplotlib | 3.8.2 | رسم المخططات |
| seaborn | 0.13.0 | تصورات بيانية |
| pandas | 2.1.4 | معالجة البيانات |

### مكتبات التطوير (اختيارية)

| المكتبة | الاستخدام |
|---------|-----------|
| pytest | الاختبارات |
| black | تنسيق الكود |
| flake8 | فحص الجودة |
| pylint | تحليل الكود |
| isort | ترتيب الاستيراد |

---

## 🎯 الخطوات التالية بعد التثبيت

1. ✅ **التحقق من التثبيت**
   ```bash
   python config.py
   ```

2. 📊 **توليد البيانات**
   ```bash
   python generate_data_enhanced.py
   ```

3. 🎓 **تدريب النموذج**
   ```bash
   python train_improved.py
   ```

4. 🧪 **اختبار النموذج**
   ```bash
   python test_model.py
   ```

5. 🚀 **تشغيل التطبيق**
   ```bash
   streamlit run app_enhanced.py
   ```

---

## 💡 نصائح مهمة

### للمبتدئين
- ✅ استخدم البيئة الافتراضية دائماً
- ✅ لا تغلق Terminal أثناء التشغيل
- ✅ اقرأ رسائل الأخطاء بعناية

### للمتقدمين
- ✅ استخدم `pip freeze > requirements.txt` بعد إضافة مكتبات
- ✅ راجع `config.py` للتخصيص
- ✅ استخدم Git لتتبع التغييرات

---

## 📞 الحصول على مساعدة

إذا واجهت مشاكل:

1. راجع قسم "حل المشاكل" أعلاه
2. تحقق من [README.md](README.md)
3. ابحث في Issues على GitHub
4. افتح Issue جديد مع تفاصيل المشكلة

---

<div align="center">

**✅ بعد التثبيت الناجح، أنت جاهز للبدء!**

انتقل إلى [QUICKSTART.md](QUICKSTART.md) للبدء السريع

</div>
