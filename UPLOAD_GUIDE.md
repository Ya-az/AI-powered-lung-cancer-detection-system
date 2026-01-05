# 📤 دليل رفع المشروع على GitHub

## الخطوة 1: تجهيز المشروع

تم تجهيز المشروع بالكامل! الملفات جاهزة للرفع.

## الخطوة 2: رفع المشروع يدوياً

### الطريقة الأولى: استخدام GitHub Desktop (الأسهل)

1. حمل وثبت GitHub Desktop من: https://desktop.github.com/
2. افتح GitHub Desktop
3. اختر: File → Add Local Repository
4. اختر مجلد المشروع: `c:\Users\Admin\OneDrive\Desktop\Lung_cancer_detector`
5. اضغط "Publish repository"
6. أدخل:
   - Repository name: `AI-powered-lung-cancer-detection-system`
   - Description: اتركه فارغاً أو أضف وصف
   - اختر Keep this code private أو اجعله Public
7. اضغط "Publish Repository"

### الطريقة الثانية: استخدام VS Code

1. افتح المشروع في VS Code
2. من القائمة الجانبية اختر Source Control (Ctrl+Shift+G)
3. اضغط "Initialize Repository"
4. اضغط "+" بجوار Changes لإضافة جميع الملفات
5. اكتب رسالة commit: "Initial commit"
6. اضغط ✓ للـ commit
7. اضغط "Publish Branch"
8. سجل دخول بحساب GitHub
9. اختر اسم المستودع: `AI-powered-lung-cancer-detection-system`
10. اضغط "Publish"

### الطريقة الثالثة: استخدام Terminal (متقدم)

افتح PowerShell في مجلد المشروع ونفذ:

```powershell
# 1. تهيئة Git
git init

# 2. إضافة جميع الملفات
git add .

# 3. إنشاء commit
git commit -m "Initial commit: AI-Powered Lung Cancer Detection System v2.0"

# 4. تسمية الفرع الرئيسي
git branch -M main

# 5. إضافة المستودع البعيد
git remote add origin https://github.com/Ya-az/AI-powered-lung-cancer-detection-system.git

# 6. رفع الملفات
git push -u origin main
```

**ملاحظة**: إذا طلب منك تسجيل دخول، استخدم:
- Username: اسم مستخدم GitHub
- Password: استخدم Personal Access Token (ليس كلمة المرور)

### كيفية إنشاء Personal Access Token:

1. اذهب إلى: https://github.com/settings/tokens
2. اضغط "Generate new token" → "Generate new token (classic)"
3. أدخل اسم: "Lung Cancer Detector"
4. اختر Expiration: 90 days
5. حدد Scopes: `repo` (كامل)
6. اضغط "Generate token"
7. انسخ الـ token (سيظهر مرة واحدة فقط!)
8. استخدمه كـ password في Git

## الخطوة 3: تفعيل GitHub Pages

بعد رفع المشروع:

1. اذهب إلى: https://github.com/Ya-az/AI-powered-lung-cancer-detection-system
2. اضغط "Settings"
3. من القائمة الجانبية اختر "Pages"
4. في قسم "Source":
   - Branch: اختر `main`
   - Folder: اختر `/ (root)`
5. اضغط "Save"
6. انتظر 2-3 دقائق
7. ستظهر رسالة: "Your site is live at https://ya-az.github.io/AI-powered-lung-cancer-detection-system/"

## الخطوة 4: اختبار الموقع

بعد تفعيل Pages:

1. افتح: https://ya-az.github.io/AI-powered-lung-cancer-detection-system/
2. ستظهر صفحة التحويل
3. انتظر ثانيتين أو اضغط الرابط
4. سيتم توجيهك إلى: https://ya-az.github.io/AI-powered-lung-cancer-detection-system/web/index.html
5. اختبر رفع صورة وتحميل تقرير PDF

## 🎉 تم بنجاح!

إذا واجهت أي مشكلة:

### المشكلة: "fatal: not a git repository"
**الحل**: نفذ `git init` أولاً

### المشكلة: "error: failed to push"
**الحل**: تأكد من:
- اتصالك بالإنترنت
- أنك صاحب المستودع
- استخدام Personal Access Token بدلاً من كلمة المرور

### المشكلة: "404 Page Not Found" على GitHub Pages
**الحل**: 
- انتظر 2-3 دقائق بعد التفعيل
- تأكد من تفعيل Pages في Settings
- تأكد من اختيار Branch: main

### المشكلة: الصفحة لا تظهر بشكل صحيح
**الحل**:
- افتح Developer Tools (F12)
- تحقق من Console للأخطاء
- تأكد من رفع مجلد `web/` كاملاً

---

## 📞 دعم إضافي

راجع [GITHUB_SETUP.md](GITHUB_SETUP.md) لمزيد من التفاصيل.

© 2026 Yazeed Aljuwaybiri
