"""
ملف التكوين المركزي لمشروع الكشف عن سرطان الرئة
يحتوي على جميع الإعدادات والمسارات والمعلمات المشتركة
"""

import os
from pathlib import Path

# ===== المسارات =====
BASE_DIR = Path(__file__).parent
IMG_DIR = BASE_DIR / "img"
MODELS_DIR = BASE_DIR / "models"

# إنشاء المجلدات إذا لم تكن موجودة
IMG_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# مسارات النماذج
MODEL_PATH_PYTORCH = str(MODELS_DIR / "lung_model.pth")
MODEL_PATH_SKLEARN = str(MODELS_DIR / "lung_model.pkl")
SCALER_PATH = str(MODELS_DIR / "scaler.pkl")

# مسارات بديلة (للتوافق مع الكود القديم)
MODEL_PATH = "lung_model.pth"
OLD_SCALER_PATH = "scaler.pkl"

# ===== إعدادات الصور =====
IMAGE_SIZE = (224, 224)
IMAGE_FORMATS = ["jpg", "jpeg", "png", "bmp", "gif"]

# إعدادات التطبيع (ImageNet)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ===== إعدادات النموذج =====
# PyTorch CNN
CNN_CONFIG = {
    "input_channels": 3,
    "conv_channels": [32, 64, 128, 256],
    "fc_sizes": [512, 256],
    "dropout_rate": 0.5,
    "num_classes": 2
}

# Scikit-learn
SKLEARN_CONFIG = {
    "n_estimators": 300,
    "learning_rate": 0.01,
    "max_depth": 8,
    "random_state": 42
}

# ===== إعدادات التدريب =====
TRAINING_CONFIG = {
    "batch_size": 4,
    "learning_rate": 0.001,
    "num_epochs": 20,
    "val_split": 0.2,
    "early_stopping_patience": 5,
    "optimizer": "adam",
    "loss_function": "cross_entropy"
}

# ===== إعدادات Data Augmentation =====
AUGMENTATION_CONFIG = {
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "horizontal_flip": True,
    "vertical_flip": False,
    "zoom_range": 0.1,
    "brightness_range": [0.8, 1.2]
}

# ===== أسماء الفئات =====
CLASS_NAMES = ["Normal", "Cancer"]
CLASS_NAMES_AR = ["سليمة", "سرطان"]
CLASS_LABELS = {
    "normal": 0,
    "healthy": 0,
    "cancer": 1,
    "tumor": 1,
    "disease": 1
}

# ===== إعدادات توليد البيانات =====
DATA_GENERATION_CONFIG = {
    "num_normal_images": 15,
    "num_cancer_images": 15,
    "image_size": IMAGE_SIZE,
    "base_intensity": 150,
    "noise_range": (-5, 5),
    "tumor_count_range": (2, 5),
    "tumor_size_range": (15, 35),
    "gaussian_blur_radius": 2
}

# ===== إعدادات Streamlit =====
STREAMLIT_CONFIG = {
    "page_title": "تصنيف أمراض الرئة",
    "page_icon": "🫁",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# ===== رسائل التطبيق =====
MESSAGES = {
    "ar": {
        "model_not_found": "⚠️ النموذج غير موجود! يرجى تدريب النموذج أولاً",
        "model_loaded": "✓ تم تحميل النموذج بنجاح",
        "upload_image": "اختر صورة أشعة للرئة",
        "analyzing": "جاري تحليل الصورة...",
        "healthy_result": "الحمد لله - الرئة سليمة",
        "cancer_result": "تحذير - قد تحتوي على سرطان",
        "medical_warning": "هذا النموذج للأغراض التعليمية فقط. استشر طبيب متخصص.",
        "processing_error": "حدث خطأ أثناء معالجة الصورة"
    },
    "en": {
        "model_not_found": "⚠️ Model not found! Please train the model first",
        "model_loaded": "✓ Model loaded successfully",
        "upload_image": "Choose a lung X-ray image",
        "analyzing": "Analyzing image...",
        "healthy_result": "Healthy Lung",
        "cancer_result": "Warning - Possible Cancer",
        "medical_warning": "This model is for educational purposes only. Consult a specialist.",
        "processing_error": "An error occurred while processing the image"
    }
}

# ===== إعدادات التسجيل (Logging) =====
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": str(BASE_DIR / "app.log")
}

# ===== إعدادات الجهاز =====
def get_device():
    """تحديد الجهاز المستخدم (CPU/GPU)"""
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return "cpu"

DEVICE = get_device()

# ===== إعدادات النشر =====
DEPLOYMENT_CONFIG = {
    "debug": True,
    "port": 8501,
    "host": "localhost",
    "max_upload_size": 10  # MB
}

# ===== متغيرات البيئة =====
import os

# يمكن تجاوز الإعدادات من متغيرات البيئة
MODEL_PATH = os.getenv("MODEL_PATH", MODEL_PATH)
IMG_DIR = Path(os.getenv("IMG_DIR", str(IMG_DIR)))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", TRAINING_CONFIG["batch_size"]))

# ===== دالات مساعدة =====
def get_model_path(model_type="pytorch"):
    """الحصول على مسار النموذج حسب النوع"""
    if model_type.lower() == "pytorch":
        return MODEL_PATH_PYTORCH if os.path.exists(MODEL_PATH_PYTORCH) else MODEL_PATH
    elif model_type.lower() == "sklearn":
        return MODEL_PATH_SKLEARN
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_message(key, language="ar"):
    """الحصول على رسالة بلغة معينة"""
    return MESSAGES.get(language, MESSAGES["ar"]).get(key, key)

# ===== معلومات المشروع =====
PROJECT_INFO = {
    "name": "Lung Cancer Detection System",
    "version": "2.0.0",
    "author": "Fahad Bandar",
    "description": "نظام ذكي للكشف عن سرطان الرئة باستخدام التعلم العميق",
    "license": "MIT",
    "year": 2026
}

if __name__ == "__main__":
    print("=" * 50)
    print(f"🫁 {PROJECT_INFO['name']} v{PROJECT_INFO['version']}")
    print("=" * 50)
    print(f"\n📁 المسارات:")
    print(f"   - مجلد المشروع: {BASE_DIR}")
    print(f"   - مجلد الصور: {IMG_DIR}")
    print(f"   - مجلد النماذج: {MODELS_DIR}")
    print(f"\n🤖 النماذج:")
    print(f"   - PyTorch: {MODEL_PATH_PYTORCH}")
    print(f"   - Scikit-learn: {MODEL_PATH_SKLEARN}")
    print(f"\n⚙️ الإعدادات:")
    print(f"   - حجم الصورة: {IMAGE_SIZE}")
    print(f"   - حجم الدُفعة: {TRAINING_CONFIG['batch_size']}")
    print(f"   - عدد العصور: {TRAINING_CONFIG['num_epochs']}")
    print(f"   - الجهاز: {DEVICE}")
    print("\n" + "=" * 50)
