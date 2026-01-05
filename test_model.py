"""
سكريبت لاختبار وتقييم نموذج الكشف عن سرطان الرئة
يعرض مصفوفة الالتباس، تقرير التصنيف، ومقاييس الأداء
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
import warnings
warnings.filterwarnings('ignore')

# محاولة استيراد PyTorch
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch غير متوفر. سيتم اختبار نموذج Scikit-learn فقط.")

# استيراد Scikit-learn
try:
    import pickle
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn غير متوفر.")

# استيراد الإعدادات
try:
    from config import *
except ImportError:
    # إعدادات افتراضية إذا لم يكن config.py موجوداً
    IMG_DIR = "img"
    MODEL_PATH = "lung_model.pth"
    MODEL_PATH_SKLEARN = "lung_model.pkl"
    SCALER_PATH = "scaler.pkl"
    IMAGE_SIZE = (224, 224)
    CLASS_NAMES_AR = ["سليمة", "سرطان"]

# ==================== PyTorch Model ====================
if PYTORCH_AVAILABLE:
    class LungClassifier(nn.Module):
        """نموذج CNN"""
        def __init__(self):
            super(LungClassifier, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
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

# ==================== Scikit-learn Functions ====================
def extract_features_sklearn(image_path):
    """استخراج ميزات للنموذج Scikit-learn"""
    from PIL import Image
    from scipy.ndimage import label
    
    img = Image.open(image_path).convert('L')
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    
    features = []
    
    # إحصائيات أساسية
    features.append(np.mean(arr))
    features.append(np.std(arr))
    features.append(np.min(arr))
    features.append(np.max(arr))
    
    # Percentiles
    features.extend([np.percentile(arr, 25), np.percentile(arr, 50), 
                     np.percentile(arr, 75), np.percentile(arr, 90)])
    
    # كشف البقع الداكنة
    thresh = np.percentile(arr, 30)
    dark = arr < thresh
    features.append(np.sum(dark))
    features.append(np.sum(dark) / arr.size)
    
    # Connected components
    try:
        labeled, num = label(dark)
        features.append(num)
        if num > 0:
            sizes = np.bincount(labeled.ravel())
            features.append(np.mean(sizes[1:]))
            features.append(np.max(sizes[1:]))
        else:
            features.extend([0, 0])
    except:
        features.extend([0, 0, 0])
    
    # كشف الحواف
    edges_x = np.abs(np.diff(arr, axis=0)).mean()
    edges_y = np.abs(np.diff(arr, axis=1)).mean()
    features.extend([edges_x, edges_y, edges_x + edges_y])
    
    # Histogram
    hist, _ = np.histogram(arr, bins=4, range=(0, 256))
    hist = hist / hist.sum()
    features.extend(hist)
    
    # تحليل المركز
    center = arr[50:174, 50:174].mean()
    features.append(center)
    features.append((arr[50:174, 50:174] < thresh).sum())
    
    # Variance
    features.append(np.var(arr - arr.mean()))
    features.append(np.sum(arr < np.percentile(arr, 20)))
    
    return np.array(features)

# ==================== Testing Functions ====================
def load_test_data():
    """تحميل بيانات الاختبار"""
    if not os.path.exists(IMG_DIR):
        print(f"❌ مجلد {IMG_DIR} غير موجود!")
        return None, None, None
    
    images_path = []
    labels = []
    
    for filename in sorted(os.listdir(IMG_DIR)):
        filepath = os.path.join(IMG_DIR, filename)
        if not os.path.isfile(filepath):
            continue
        
        if "normal" in filename.lower():
            label = 0
        elif "cancer" in filename.lower():
            label = 1
        else:
            continue
        
        images_path.append(filepath)
        labels.append(label)
    
    print(f"✓ تم العثور على {len(images_path)} صورة")
    print(f"  - سليمة: {labels.count(0)}")
    print(f"  - سرطان: {labels.count(1)}")
    
    return images_path, np.array(labels), None

def test_pytorch_model(images_path, true_labels):
    """اختبار نموذج PyTorch"""
    if not PYTORCH_AVAILABLE:
        print("⚠️ PyTorch غير متوفر")
        return None, None
    
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ النموذج {MODEL_PATH} غير موجود")
        return None, None
    
    print("\n" + "="*60)
    print("🧪 اختبار نموذج PyTorch")
    print("="*60)
    
    # تحميل النموذج
    device = torch.device("cpu")
    model = LungClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # التنبؤ
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    predictions = []
    probabilities = []
    
    print("\n🔄 جاري التنبؤ...")
    for img_path in images_path:
        image = Image.open(img_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            
            predictions.append(pred)
            probabilities.append(probs[0].cpu().numpy())
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    return predictions, probabilities

def test_sklearn_model(images_path, true_labels):
    """اختبار نموذج Scikit-learn"""
    if not SKLEARN_AVAILABLE:
        print("⚠️ Scikit-learn غير متوفر")
        return None, None
    
    model_path = MODEL_PATH_SKLEARN if os.path.exists(MODEL_PATH_SKLEARN) else "lung_model.pkl"
    scaler_path = SCALER_PATH if os.path.exists(SCALER_PATH) else "scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"⚠️ النموذج أو Scaler غير موجود")
        return None, None
    
    print("\n" + "="*60)
    print("🧪 اختبار نموذج Scikit-learn")
    print("="*60)
    
    # تحميل النموذج
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # استخراج الميزات
    print("\n🔄 جاري استخراج الميزات...")
    features_list = []
    for img_path in images_path:
        features = extract_features_sklearn(img_path)
        features_list.append(features)
    
    X = np.array(features_list)
    X_scaled = scaler.transform(X)
    
    # التنبؤ
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    return predictions, probabilities

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """رسم مصفوفة الالتباس"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES_AR, 
                yticklabels=CLASS_NAMES_AR,
                cbar_kws={'label': 'Count'})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('القيمة الفعلية', fontsize=12, fontweight='bold')
    plt.xlabel('القيمة المتوقعة', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    return cm

def plot_metrics(y_true, y_pred, y_proba, model_name):
    """رسم المقاييس المختلفة"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'تقييم أداء نموذج {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=CLASS_NAMES_AR, yticklabels=CLASS_NAMES_AR)
    axes[0, 0].set_title('مصفوفة الالتباس', fontweight='bold')
    axes[0, 0].set_ylabel('القيمة الفعلية')
    axes[0, 0].set_xlabel('القيمة المتوقعة')
    
    # 2. Performance Metrics Bar Chart
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1-Score': f1_score(y_true, y_pred, average='weighted')
    }
    
    bars = axes[0, 1].bar(metrics.keys(), metrics.values(), 
                          color=['#28a745', '#dc3545', '#ffc107', '#17a2b8'],
                          alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].set_title('مقاييس الأداء', fontweight='bold')
    axes[0, 1].set_ylabel('القيمة')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # إضافة القيم على الأعمدة
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. ROC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                       label='Random')
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel('معدل الإيجابيات الخاطئة')
        axes[1, 0].set_ylabel('معدل الإيجابيات الصحيحة')
        axes[1, 0].set_title('منحنى ROC', fontweight='bold')
        axes[1, 0].legend(loc="lower right")
        axes[1, 0].grid(alpha=0.3)
    
    # 4. Class Distribution
    unique, counts = np.unique(y_true, return_counts=True)
    axes[1, 1].pie(counts, labels=[CLASS_NAMES_AR[i] for i in unique], 
                   autopct='%1.1f%%', startangle=90,
                   colors=['#28a745', '#dc3545'])
    axes[1, 1].set_title('توزيع الفئات', fontweight='bold')
    
    plt.tight_layout()
    return fig

def print_detailed_report(y_true, y_pred, y_proba, model_name):
    """طباعة تقرير تفصيلي"""
    print("\n" + "="*60)
    print(f"📊 تقرير التقييم التفصيلي - {model_name}")
    print("="*60)
    
    # المقاييس الأساسية
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n✅ الدقة (Accuracy):       {accuracy*100:.2f}%")
    print(f"🎯 الدقة (Precision):      {precision*100:.2f}%")
    print(f"📈 الاستدعاء (Recall):     {recall*100:.2f}%")
    print(f"⚖️  F1-Score:              {f1*100:.2f}%")
    
    # مصفوفة الالتباس
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📋 مصفوفة الالتباس:")
    print(f"   {'':12} {'سليمة':>10} {'سرطان':>10}")
    print(f"   {'سليمة':12} {cm[0,0]:10d} {cm[0,1]:10d}")
    print(f"   {'سرطان':12} {cm[1,0]:10d} {cm[1,1]:10d}")
    
    # تقرير التصنيف
    print(f"\n📝 تقرير التصنيف:")
    report = classification_report(y_true, y_pred, 
                                   target_names=CLASS_NAMES_AR,
                                   digits=3)
    print(report)
    
    # ROC AUC
    if y_proba is not None:
        from sklearn.metrics import roc_auc_score
        try:
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
            print(f"\n🔍 ROC AUC Score:          {roc_auc:.4f}")
        except:
            pass

def main():
    """الدالة الرئيسية"""
    print("\n" + "="*60)
    print("🫁 نظام اختبار نموذج الكشف عن سرطان الرئة")
    print("="*60)
    
    # تحميل البيانات
    images_path, true_labels, _ = load_test_data()
    
    if images_path is None or len(images_path) == 0:
        print("❌ لا توجد بيانات للاختبار!")
        return
    
    # اختبار PyTorch
    if PYTORCH_AVAILABLE and os.path.exists(MODEL_PATH):
        predictions_pt, probabilities_pt = test_pytorch_model(images_path, true_labels)
        
        if predictions_pt is not None:
            print_detailed_report(true_labels, predictions_pt, probabilities_pt, "PyTorch CNN")
            fig = plot_metrics(true_labels, predictions_pt, probabilities_pt, "PyTorch CNN")
            plt.savefig("pytorch_evaluation.png", dpi=300, bbox_inches='tight')
            print("\n💾 تم حفظ الرسوم البيانية في pytorch_evaluation.png")
            plt.show()
    
    # اختبار Scikit-learn
    model_path = MODEL_PATH_SKLEARN if os.path.exists(MODEL_PATH_SKLEARN) else "lung_model.pkl"
    if SKLEARN_AVAILABLE and os.path.exists(model_path):
        predictions_sk, probabilities_sk = test_sklearn_model(images_path, true_labels)
        
        if predictions_sk is not None:
            print_detailed_report(true_labels, predictions_sk, probabilities_sk, "Scikit-learn")
            fig = plot_metrics(true_labels, predictions_sk, probabilities_sk, "Scikit-learn")
            plt.savefig("sklearn_evaluation.png", dpi=300, bbox_inches='tight')
            print("\n💾 تم حفظ الرسوم البيانية في sklearn_evaluation.png")
            plt.show()
    
    print("\n" + "="*60)
    print("✅ اكتمل الاختبار بنجاح!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
