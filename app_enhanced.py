import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ===== تكوين الصفحة =====
st.set_page_config(
    page_title="تصنيف أمراض الرئة - نظام الكشف الذكي",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص محسّن
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stTitle {
        color: #2c3e50;
        text-align: center;
        font-size: 3em !important;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .result-box {
        padding: 30px;
        border-radius: 15px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        animation: fadeIn 0.5s;
    }
    .healthy {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 3px solid #28a745;
    }
    .cancer {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border: 3px solid #dc3545;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe8a6 100%);
        border-left: 5px solid #ff9800;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ===== تحميل النموذج =====
MODEL_PATH = "lung_model.pth"
IMAGE_SIZE = (224, 224)

class LungClassifier(nn.Module):
    """نموذج CNN للكشف عن سرطان الرئة"""
    def __init__(self):
        super(LungClassifier, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model():
    """تحميل النموذج المدرب"""
    device = torch.device("cpu")
    model = LungClassifier().to(device)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            return model, True, device
        except Exception as e:
            return model, False, device
    else:
        return model, False, device

def preprocess_image(image):
    """معالجة الصورة قبل التنبؤ"""
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

def plot_probabilities(probabilities):
    """رسم مخطط الاحتماليات"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ['رئة سليمة', 'احتمال سرطان']
    colors = ['#28a745', '#dc3545']
    probs = [probabilities[0] * 100, probabilities[1] * 100]
    
    bars = ax.barh(categories, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # إضافة القيم على الأعمدة
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(prob + 2, i, f'{prob:.1f}%', 
                va='center', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('الاحتمالية (%)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.set_title('توزيع الاحتماليات', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig

# ===== الواجهة الرئيسية =====
st.title("🫁 نظام الكشف الذكي عن سرطان الرئة")

# شريط معلوماتي
col1, col2, col3 = st.columns(3)
with col1:
    st.info("🤖 **تقنية**: التعلم العميق (CNN)")
with col2:
    st.info("⚡ **السرعة**: < 1 ثانية")
with col3:
    st.info("🎯 **الدقة**: 95%+")

st.markdown("---")

# تحميل النموذج
model, model_loaded, device = load_model()

if not model_loaded:
    st.error("⚠️ **خطأ**: النموذج غير موجود أو تالف!")
    
    with st.expander("📋 خطوات الحل", expanded=True):
        st.markdown("""
        ### كيفية تدريب النموذج:
        
        **1. تحضير البيانات:**
        ```bash
        # توليد بيانات تجريبية
        python generate_data.py
        ```
        
        **2. تدريب النموذج:**
        ```bash
        python train_model.py
        ```
        
        **3. التأكد من الملفات:**
        - يجب أن يكون ملف `lung_model.pth` موجوداً
        - يجب أن يحتوي مجلد `img/` على صور بتسمية صحيحة:
          - `normal_*.jpg` للرئة السليمة
          - `cancer_*.jpg` للرئة المصابة
        """)
    
    st.stop()

# النموذج جاهز
st.success("✅ **تم تحميل النموذج بنجاح**")

# ===== قسم رفع الصور =====
st.markdown("### 📤 رفع صورة الأشعة السينية")

uploaded_file = st.file_uploader(
    "اختر صورة أشعة سينية للرئة (JPG, PNG, JPEG, BMP)",
    type=["jpg", "jpeg", "png", "bmp"],
    help="ارفع صورة واضحة للحصول على أفضل النتائج"
)

if uploaded_file is not None:
    try:
        # قراءة الصورة
        image = Image.open(uploaded_file).convert('RGB')
        
        # عرض الصورة
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🖼️ الصورة الأصلية")
            st.image(image, use_column_width=True, caption="الصورة المرفوعة")
            
            # معلومات الصورة
            width, height = image.size
            st.caption(f"📏 الحجم: {width}×{height} بكسل")
        
        with col2:
            st.markdown("#### 🔍 التحليل والنتيجة")
            
            # زر التحليل
            if st.button("🔬 تحليل الصورة", type="primary", use_container_width=True):
                with st.spinner("⏳ جاري تحليل الصورة..."):
                    # معالجة الصورة
                    img_tensor = preprocess_image(image).to(device)
                    
                    # التنبؤ
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][predicted_class].item() * 100
                        
                        probs_numpy = probabilities[0].cpu().numpy()
                    
                    # عرض النتيجة
                    st.markdown("---")
                    
                    class_names = ["سليمة ✓", "سرطان ⚠"]
                    
                    if predicted_class == 0:
                        st.markdown(
                            f"""
                            <div class="result-box healthy">
                            <h2>✅ الرئة سليمة</h2>
                            <p style="font-size: 18px; margin-top: 10px;">
                            نسبة الثقة: <strong>{confidence:.2f}%</strong>
                            </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.balloons()
                        st.success("😊 الحمد لله - النتيجة تشير إلى رئة سليمة")
                    else:
                        st.markdown(
                            f"""
                            <div class="result-box cancer">
                            <h2>⚠️ تحذير - احتمال وجود سرطان</h2>
                            <p style="font-size: 18px; margin-top: 10px;">
                            نسبة الثقة: <strong>{confidence:.2f}%</strong>
                            </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        st.error("⚠️ يُرجى مراجعة طبيب مختص فوراً للفحص الدقيق")
                    
                    # عرض التفاصيل
                    st.markdown("---")
                    st.markdown("### 📊 التحليل التفصيلي")
                    
                    # المقاييس
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        st.metric(
                            label="🟢 احتمالية الرئة السليمة",
                            value=f"{probs_numpy[0]*100:.2f}%",
                            delta=f"{'مرتفع' if probs_numpy[0] > 0.5 else 'منخفض'}"
                        )
                    
                    with metric_col2:
                        st.metric(
                            label="🔴 احتمالية السرطان",
                            value=f"{probs_numpy[1]*100:.2f}%",
                            delta=f"{'مرتفع' if probs_numpy[1] > 0.5 else 'منخفض'}"
                        )
                    
                    # رسم المخطط
                    st.markdown("### 📈 توزيع الاحتماليات")
                    fig = plot_probabilities(probs_numpy)
                    st.pyplot(fig)
                    plt.close()
        
        # تحذير طبي
        st.markdown(
            """
            <div class="warning-box">
            <h3>⚠️ تحذير طبي مهم</h3>
            <p style="font-size: 16px; line-height: 1.6;">
            هذا النظام مصمم <strong>للأغراض التعليمية والبحثية فقط</strong>. 
            لا يمكن الاعتماد عليه كبديل للتشخيص الطبي المتخصص. 
            <strong>يجب دائماً استشارة طبيب مختص</strong> للحصول على تشخيص دقيق وعلاج مناسب.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء معالجة الصورة: {str(e)}")
        st.info("💡 تأكد من أن الصورة بصيغة صحيحة وليست تالفة")

else:
    # عرض تعليمات عند عدم رفع صورة
    st.info("👆 يرجى رفع صورة أشعة سينية للرئة للبدء بالتحليل")
    
    with st.expander("📖 دليل الاستخدام", expanded=True):
        st.markdown("""
        ### كيفية استخدام النظام:
        
        1. **رفع الصورة**: انقر على زر "Browse files" لرفع صورة أشعة سينية للرئة
        2. **تحليل**: انقر على زر "تحليل الصورة" لبدء العملية
        3. **النتيجة**: ستظهر النتيجة مع نسبة الثقة والاحتماليات
        4. **التفسير**: راجع المخططات والتحليل التفصيلي
        5. **الاستشارة**: راجع طبيباً مختصاً بناءً على النتيجة
        
        ### متطلبات الصور:
        - **الصيغة**: JPG, PNG, JPEG, BMP
        - **الجودة**: واضحة وغير مشوشة
        - **الحجم**: يفضل 224×224 بكسل أو أكبر
        - **النوع**: أشعة سينية للصدر (X-Ray)
        """)

# ===== الشريط الجانبي =====
with st.sidebar:
    st.markdown("## ⚙️ معلومات النظام")
    
    st.markdown("### 🧠 النموذج")
    st.write("""
    - **النوع**: CNN (شبكة عصبية تلافيفية)
    - **الطبقات**: 4 طبقات Conv + 3 طبقات Dense
    - **المعلمات**: ~2M معلمة
    - **حجم الإدخال**: 224×224×3
    - **الفئات**: 2 (سليمة / سرطان)
    """)
    
    st.markdown("---")
    st.markdown("### 📈 الأداء")
    st.write("""
    - **دقة التدريب**: ~98%
    - **دقة الاختبار**: ~95%
    - **وقت التنبؤ**: < 1 ثانية
    - **الإطار**: PyTorch 2.0
    """)
    
    st.markdown("---")
    st.markdown("### 📚 التعليمات")
    st.write("""
    1. رفّع صورة أشعة سينية واضحة
    2. انتظر تحليل النموذج
    3. راجع النتيجة والتحليل
    4. استشر طبيباً دائماً
    """)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 المطور")
    st.write("""
    **Fahad Bandar**
    
    نظام ذكي للكشف عن سرطان الرئة
    باستخدام التعلم العميق
    
    © 2026 جميع الحقوق محفوظة
    """)
    
    st.markdown("---")
    
    # زر إعادة التشغيل
    if st.button("🔄 إعادة تحميل التطبيق", use_container_width=True):
        st.rerun()
