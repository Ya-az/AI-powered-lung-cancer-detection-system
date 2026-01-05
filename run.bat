@echo off
REM ==================================================
REM سكريبت التشغيل السريع لمشروع الكشف عن سرطان الرئة
REM ==================================================

echo.
echo ========================================
echo    نظام الكشف عن سرطان الرئة
echo    Lung Cancer Detection System
echo ========================================
echo.

:menu
echo.
echo اختر احد الخيارات:
echo.
echo [1] تثبيت المكتبات المطلوبة
echo [2] توليد بيانات تجريبية
echo [3] توليد بيانات محسنة
echo [4] تدريب نموذج PyTorch
echo [5] تدريب نموذج Scikit-learn
echo [6] اختبار النموذج
echo [7] تشغيل تطبيق Streamlit (PyTorch)
echo [8] تشغيل تطبيق Streamlit المحسن
echo [9] تشغيل تطبيق Streamlit (Sklearn)
echo [0] فتح واجهة HTML
echo [X] خروج
echo.

set /p choice="اختر رقم الخيار: "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto generate
if "%choice%"=="3" goto generate_enhanced
if "%choice%"=="4" goto train_pytorch
if "%choice%"=="5" goto train_sklearn
if "%choice%"=="6" goto test
if "%choice%"=="7" goto run_app
if "%choice%"=="8" goto run_app_enhanced
if "%choice%"=="9" goto run_sklearn
if "%choice%"=="0" goto open_html
if /i "%choice%"=="x" goto end
goto menu

:install
echo.
echo ========================================
echo تثبيت المكتبات المطلوبة...
echo ========================================
pip install -r requirements.txt
echo.
echo تم التثبيت بنجاح!
pause
goto menu

:generate
echo.
echo ========================================
echo توليد بيانات تجريبية...
echo ========================================
python generate_data.py
echo.
pause
goto menu

:generate_enhanced
echo.
echo ========================================
echo توليد بيانات محسنة...
echo ========================================
python generate_data_enhanced.py
echo.
pause
goto menu

:train_pytorch
echo.
echo ========================================
echo تدريب نموذج PyTorch...
echo ========================================
python train_model.py
echo.
pause
goto menu

:train_sklearn
echo.
echo ========================================
echo تدريب نموذج Scikit-learn...
echo ========================================
python train_improved.py
echo.
pause
goto menu

:test
echo.
echo ========================================
echo اختبار النموذج...
echo ========================================
python test_model.py
echo.
pause
goto menu

:run_app
echo.
echo ========================================
echo تشغيل تطبيق Streamlit (PyTorch)...
echo ========================================
echo.
echo سيتم فتح المتصفح تلقائياً على http://localhost:8501
echo للإيقاف اضغط Ctrl+C
echo.
streamlit run app.py
pause
goto menu

:run_app_enhanced
echo.
echo ========================================
echo تشغيل تطبيق Streamlit المحسن...
echo ========================================
echo.
echo سيتم فتح المتصفح تلقائياً على http://localhost:8501
echo للإيقاف اضغط Ctrl+C
echo.
streamlit run app_enhanced.py
pause
goto menu

:run_sklearn
echo.
echo ========================================
echo تشغيل تطبيق Streamlit (Sklearn)...
echo ========================================
echo.
echo سيتم فتح المتصفح تلقائياً على http://localhost:8501
echo للإيقاف اضغط Ctrl+C
echo.
streamlit run app_sklearn.py
pause
goto menu

:open_html
echo.
echo ========================================
echo فتح واجهة HTML...
echo ========================================
start index.html
pause
goto menu

:end
echo.
echo شكراً لاستخدامك النظام!
echo.
exit
