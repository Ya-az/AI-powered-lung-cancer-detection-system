# ========================================
# Project Organization Script
# Organizes files into proper folders
# ========================================

Write-Host "🔧 Organizing Lung Cancer Detector Project..." -ForegroundColor Cyan

# Navigate to project root
$projectRoot = "C:\Users\Admin\OneDrive\Desktop\Lung_cancer_detector"
Set-Location $projectRoot

# Python files to src/
Write-Host "`n📦 Moving Python files to src/..." -ForegroundColor Yellow
$pyFiles = @(
    "app.py", "app_enhanced.py", "app_simple.py", "app_sklearn.py",
    "generate_data.py", "generate_data_enhanced.py",
    "train_model.py", "train_improved.py", "train_model_sklearn.py",
    "test_model.py"
)
foreach ($file in $pyFiles) {
    if (Test-Path $file) {
        Copy-Item $file "src\" -Force
        Write-Host "  ✓ $file" -ForegroundColor Green
    }
}

# Web files to web/
Write-Host "`n🌐 Moving web files to web/..." -ForegroundColor Yellow
$webFiles = @("index.html", "index_enhanced.html")
foreach ($file in $webFiles) {
    if (Test-Path $file) {
        Copy-Item $file "web\" -Force
        Write-Host "  ✓ $file" -ForegroundColor Green
    }
}

# Model files to models/
Write-Host "`n🤖 Moving model files to models/..." -ForegroundColor Yellow
$modelFiles = @("lung_model.pkl", "scaler.pkl")
foreach ($file in $modelFiles) {
    if (Test-Path $file) {
        Copy-Item $file "models\" -Force
        Write-Host "  ✓ $file" -ForegroundColor Green
    }
}

# Documentation to docs/
Write-Host "`n📚 Moving documentation to docs/..." -ForegroundColor Yellow
$docFiles = @("CHANGELOG.md", "INSTALLATION.md", "PROJECT_SUMMARY.md", "QUICKSTART.md")
foreach ($file in $docFiles) {
    if (Test-Path $file) {
        Copy-Item $file "docs\" -Force
        Write-Host "  ✓ $file" -ForegroundColor Green
    }
}

Write-Host "`n✅ Project organization complete!" -ForegroundColor Green
Write-Host "`n📁 Final structure:" -ForegroundColor Cyan
Write-Host "  src/       → Python source files" -ForegroundColor White
Write-Host "  web/       → HTML/JS interface" -ForegroundColor White
Write-Host "  models/    → Trained ML models" -ForegroundColor White
Write-Host "  docs/      → Documentation" -ForegroundColor White
Write-Host "  img/       → Sample images" -ForegroundColor White

Write-Host "`n🚀 Ready for GitHub upload!" -ForegroundColor Green
Write-Host "See GITHUB_SETUP.md for instructions." -ForegroundColor Yellow
