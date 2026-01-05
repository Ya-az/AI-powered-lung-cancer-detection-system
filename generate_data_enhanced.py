"""
سكريبت محسّن لتوليد بيانات تدريب تجريبية
ينشئ صور أشعة سينية اصطناعية للرئة (سليمة ومصابة بالسرطان)
"""

import os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import shutil

# الإعدادات
IMG_DIR = "img"
NUM_NORMAL = 20  # عدد الصور السليمة
NUM_CANCER = 20  # عدد الصور المصابة
IMAGE_SIZE = (224, 224)

print("=" * 70)
print("🫁 مولّد بيانات تدريب الكشف عن سرطان الرئة")
print("=" * 70)

# حذف المجلد القديم وإنشاء مجلد جديد
if os.path.exists(IMG_DIR):
    print(f"\n🗑️  حذف المجلد القديم {IMG_DIR}...")
    shutil.rmtree(IMG_DIR)

os.makedirs(IMG_DIR, exist_ok=True)
print(f"✅ تم إنشاء مجلد {IMG_DIR}")

# تحديد البذرة للحصول على نتائج متسقة
np.random.seed(42)

def add_lung_structure(arr):
    """إضافة بنية رئوية واقعية"""
    h, w = arr.shape
    center_x, center_y = h // 2, w // 2
    
    # إضافة تدرج من المركز
    for x in range(h):
        for y in range(w):
            # المسافة من المركز
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # تطبيق تدرج
            gradient_factor = 1 - (dist / (h * 0.7))
            gradient_factor = np.clip(gradient_factor, 0, 1)
            
            # تعديل الشدة
            arr[x, y] = arr[x, y] * (0.7 + 0.3 * gradient_factor)
    
    return arr

def add_rib_shadows(img_array):
    """إضافة ظلال الضلوع"""
    h, w = img_array.shape
    
    # إضافة خطوط أفقية تمثل الضلوع
    num_ribs = 8
    for i in range(num_ribs):
        y_pos = int(h * (0.2 + 0.6 * i / num_ribs))
        thickness = np.random.randint(2, 5)
        
        for t in range(-thickness, thickness + 1):
            if 0 <= y_pos + t < h:
                # تأثير موجي للضلع
                for x in range(w):
                    wave = int(10 * np.sin(x * np.pi / w * 2))
                    y = y_pos + t + wave
                    if 0 <= y < h:
                        img_array[y, x] = img_array[y, x] * 0.85
    
    return img_array

def add_noise(arr, intensity=10):
    """إضافة ضوضاء واقعية"""
    noise = np.random.normal(0, intensity, arr.shape)
    arr = arr + noise
    return np.clip(arr, 0, 255)

def create_tumor(size, irregular=True):
    """إنشاء ورم سرطاني واقعي"""
    tumor = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    
    for x in range(size):
        for y in range(size):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            
            if irregular:
                # جعل الحواف غير منتظمة
                angle = np.arctan2(y - center, x - center)
                variation = np.random.uniform(0.7, 1.3) * np.sin(angle * 4)
                radius = (size / 2) * (0.8 + 0.2 * variation)
            else:
                radius = size / 2
            
            if dist < radius:
                # تدرج في الكثافة
                intensity = int(255 * (1 - dist / radius) * 0.4)
                tumor[x, y] = intensity
    
    return tumor

# ==================== توليد صور الرئة السليمة ====================
print(f"\n{'='*70}")
print(f"🟢 توليد صور الرئة السليمة ({NUM_NORMAL} صورة)")
print(f"{'='*70}")

for i in range(NUM_NORMAL):
    # إنشاء صورة أساسية
    base_intensity = np.random.randint(140, 160)
    arr = np.ones(IMAGE_SIZE, dtype=np.float32) * base_intensity
    
    # إضافة بنية الرئة
    arr = add_lung_structure(arr)
    
    # إضافة ظلال الضلوع
    arr = add_rib_shadows(arr)
    
    # إضافة ضوضاء
    arr = add_noise(arr, intensity=8)
    
    # إضافة بعض التباين الطبيعي
    arr = np.clip(arr, 80, 220)
    
    # تحويل إلى صورة
    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr, mode='L')
    
    # تطبيق تنعيم
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    # تحسين التباين قليلاً
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    
    # حفظ الصورة
    filename = f'{IMG_DIR}/normal_{i+1:02d}.jpg'
    img.save(filename, quality=95)
    
    if (i + 1) % 5 == 0:
        print(f"  ✓ تم إنشاء {i+1} صورة")

print(f"  ✅ اكتمل: {NUM_NORMAL} صورة سليمة")

# ==================== توليد صور الرئة المصابة ====================
print(f"\n{'='*70}")
print(f"🔴 توليد صور الرئة المصابة بالسرطان ({NUM_CANCER} صورة)")
print(f"{'='*70}")

for i in range(NUM_CANCER):
    # إنشاء صورة أساسية مشابهة للسليمة
    base_intensity = np.random.randint(140, 160)
    arr = np.ones(IMAGE_SIZE, dtype=np.float32) * base_intensity
    
    # إضافة بنية الرئة
    arr = add_lung_structure(arr)
    
    # إضافة ظلال الضلوع
    arr = add_rib_shadows(arr)
    
    # إضافة ضوضاء
    arr = add_noise(arr, intensity=8)
    
    arr = np.clip(arr, 80, 220).astype(np.uint8)
    
    # تحويل إلى صورة PIL لإضافة الأورام
    img = Image.fromarray(arr, mode='L')
    draw = ImageDraw.Draw(img)
    
    # إضافة أورام (بقع داكنة)
    num_tumors = np.random.randint(2, 5)
    
    for _ in range(num_tumors):
        # موقع عشوائي (تجنب الحواف)
        tumor_x = np.random.randint(40, IMAGE_SIZE[0] - 40)
        tumor_y = np.random.randint(40, IMAGE_SIZE[1] - 40)
        tumor_size = np.random.randint(15, 40)
        
        # إنشاء ورم
        tumor = create_tumor(tumor_size, irregular=True)
        
        # لصق الورم على الصورة
        tumor_img = Image.fromarray(tumor, mode='L')
        
        # دمج الورم مع الصورة الأصلية
        img_array = np.array(img)
        tumor_array = np.array(tumor_img)
        
        x_start = max(0, tumor_x - tumor_size // 2)
        y_start = max(0, tumor_y - tumor_size // 2)
        x_end = min(IMAGE_SIZE[0], x_start + tumor_size)
        y_end = min(IMAGE_SIZE[1], y_start + tumor_size)
        
        tumor_h = x_end - x_start
        tumor_w = y_end - y_start
        
        if tumor_h > 0 and tumor_w > 0:
            # تطبيق الورم بطريقة مزج
            region = img_array[x_start:x_end, y_start:y_end]
            tumor_region = tumor_array[:tumor_h, :tumor_w]
            
            # جعل المنطقة أغمق
            blended = region * 0.5 + tumor_region * 0.3
            img_array[x_start:x_end, y_start:y_end] = blended
        
        img = Image.fromarray(img_array.astype(np.uint8), mode='L')
    
    # تطبيق تنعيم
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    # تحسين التباين
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.15)
    
    # حفظ الصورة
    filename = f'{IMG_DIR}/cancer_{i+1:02d}.jpg'
    img.save(filename, quality=95)
    
    if (i + 1) % 5 == 0:
        print(f"  ✓ تم إنشاء {i+1} صورة")

print(f"  ✅ اكتمل: {NUM_CANCER} صورة مصابة")

# ==================== ملخص ====================
print(f"\n{'='*70}")
print(f"📊 ملخص التوليد")
print(f"{'='*70}")
print(f"  ✅ إجمالي الصور المولدة: {NUM_NORMAL + NUM_CANCER}")
print(f"  🟢 صور سليمة: {NUM_NORMAL}")
print(f"  🔴 صور مصابة: {NUM_CANCER}")
print(f"  📁 الموقع: {os.path.abspath(IMG_DIR)}/")
print(f"  📏 الحجم: {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]} بكسل")
print(f"\n{'='*70}")
print(f"✅ تم إنشاء قاعدة البيانات بنجاح!")
print(f"{'='*70}")

print("\n💡 الخطوة التالية:")
print("   قم بتشغيل أحد أوامر التدريب التالية:")
print("   • python train_model.py        (نموذج PyTorch)")
print("   • python train_improved.py     (نموذج PyTorch محسّن)")
print("   • python train_model_sklearn.py (نموذج Scikit-learn)")
print()
