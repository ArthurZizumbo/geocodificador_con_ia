@echo off
REM setup-gcs-backend.bat
REM Script para configurar un bucket de Google Cloud Storage como backend de Terraform
REM para el proyecto de geocodificación del INE

setlocal enabledelayedexpansion

REM Variables de configuración
if "%PROJECT_ID%"=="" set PROJECT_ID=proyecto-ine-geocodificador
if "%BUCKET_NAME%"=="" set BUCKET_NAME=terraform-state-ine-geocodificador
if "%REGION%"=="" set REGION=us-central1
if "%SERVICE_ACCOUNT_NAME%"=="" set SERVICE_ACCOUNT_NAME=terraform-backend-sa
set SERVICE_ACCOUNT_EMAIL=%SERVICE_ACCOUNT_NAME%@%PROJECT_ID%.iam.gserviceaccount.com

echo 🚀 Configurando backend de Terraform en Google Cloud Storage
echo    Proyecto: %PROJECT_ID%
echo    Bucket: %BUCKET_NAME%
echo    Región: %REGION%
echo.

REM Verificar que gcloud esté instalado
gcloud version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: gcloud CLI no está instalado
    echo    Instala Google Cloud SDK: https://cloud.google.com/sdk/docs/install
    exit /b 1
)

REM Verificar autenticación
gcloud auth list --filter=status:ACTIVE --format="value(account)" | findstr "@" >nul
if errorlevel 1 (
    echo ❌ Error: No hay cuentas autenticadas en gcloud
    echo    Ejecuta: gcloud auth login
    exit /b 1
)

REM Configurar el proyecto
echo ⚙️  Configurando proyecto de Google Cloud...
gcloud config set project %PROJECT_ID%

REM Habilitar APIs necesarias
echo 🔧 Habilitando APIs necesarias...
gcloud services enable storage.googleapis.com cloudresourcemanager.googleapis.com iam.googleapis.com

REM Crear el bucket para el estado de Terraform
echo 📦 Creando bucket de GCS para el estado de Terraform...
gsutil ls -b gs://%BUCKET_NAME% >nul 2>&1
if not errorlevel 1 (
    echo    ⚠️  El bucket %BUCKET_NAME% ya existe
) else (
    gsutil mb -p %PROJECT_ID% -c STANDARD -l %REGION% gs://%BUCKET_NAME%
    echo    ✅ Bucket %BUCKET_NAME% creado exitosamente
)

REM Configurar versionado y encriptación del bucket
echo 🔒 Configurando seguridad del bucket...
gsutil versioning set on gs://%BUCKET_NAME%
gsutil uniformbucketlevelaccess set on gs://%BUCKET_NAME%

REM Crear cuenta de servicio para Terraform
echo 👤 Configurando cuenta de servicio para Terraform...
gcloud iam service-accounts describe %SERVICE_ACCOUNT_EMAIL% >nul 2>&1
if not errorlevel 1 (
    echo    ⚠️  La cuenta de servicio %SERVICE_ACCOUNT_NAME% ya existe
) else (
    gcloud iam service-accounts create %SERVICE_ACCOUNT_NAME% --display-name="Terraform Backend Service Account" --description="Cuenta de servicio para gestionar el estado de Terraform"
    echo    ✅ Cuenta de servicio %SERVICE_ACCOUNT_NAME% creada
)

REM Asignar permisos mínimos necesarios
echo 🔐 Asignando permisos a la cuenta de servicio...
gcloud projects add-iam-policy-binding %PROJECT_ID% --member="serviceAccount:%SERVICE_ACCOUNT_EMAIL%" --role="roles/storage.admin"
gcloud projects add-iam-policy-binding %PROJECT_ID% --member="serviceAccount:%SERVICE_ACCOUNT_EMAIL%" --role="roles/compute.admin"
gcloud projects add-iam-policy-binding %PROJECT_ID% --member="serviceAccount:%SERVICE_ACCOUNT_EMAIL%" --role="roles/iam.serviceAccountUser"

REM Crear y descargar clave de la cuenta de servicio
echo 🔑 Creando clave de la cuenta de servicio...
set CREDENTIALS_FILE=terraform-sa-key.json
if exist %CREDENTIALS_FILE% (
    echo    ⚠️  El archivo de credenciales ya existe: %CREDENTIALS_FILE%
    echo    ℹ️  Si necesitas regenerar las credenciales, elimina el archivo primero
) else (
    gcloud iam service-accounts keys create %CREDENTIALS_FILE% --iam-account=%SERVICE_ACCOUNT_EMAIL%
    echo    ✅ Credenciales guardadas en: %CREDENTIALS_FILE%
    echo    ⚠️  IMPORTANTE: Mantén este archivo seguro y no lo commits al repositorio
)

REM Crear archivo de configuración de backend para Terraform
echo 📝 Creando configuración de backend...
(
echo # backend-config.hcl
echo # Configuración de backend para Terraform usando Google Cloud Storage
echo.
echo bucket = "%BUCKET_NAME%"
echo prefix = "terraform/state"
) > backend-config.hcl

echo    ✅ Configuración guardada en: backend-config.hcl

REM Crear archivo de variables de entorno para Windows
echo 🌐 Creando archivo de variables de entorno...
(
echo REM Variables de entorno para Terraform
echo set GOOGLE_APPLICATION_CREDENTIALS=./terraform-sa-key.json
echo set GOOGLE_PROJECT=%PROJECT_ID%
echo set GOOGLE_REGION=%REGION%
echo set TF_VAR_project_id=%PROJECT_ID%
echo set TF_VAR_region=%REGION%
) > set-terraform-env.bat

echo    ✅ Variables de entorno guardadas en: set-terraform-env.bat

REM Crear también archivo .gitignore si no existe
if not exist .gitignore (
    (
    echo # Terraform
    echo *.tfstate
    echo *.tfstate.*
    echo .terraform/
    echo .terraform.lock.hcl
    echo terraform-sa-key.json
    echo set-terraform-env.bat
    echo .env.terraform
    ) > .gitignore
    echo    ✅ Archivo .gitignore creado
)

REM Resumen final
echo.
echo ✅ ¡Configuración completada!
echo.
echo 📋 Resumen:
echo    • Bucket de GCS: gs://%BUCKET_NAME%
echo    • Cuenta de servicio: %SERVICE_ACCOUNT_EMAIL%
echo    • Credenciales: %CREDENTIALS_FILE%
echo    • Configuración backend: backend-config.hcl
echo    • Variables de entorno: set-terraform-env.bat
echo.
echo 🚀 Próximos pasos:
echo    1. Carga las variables de entorno: set-terraform-env.bat
echo    2. Inicializa Terraform: terraform init -backend-config=backend-config.hcl
echo    3. Verifica que %CREDENTIALS_FILE% esté en .gitignore
echo.
echo ⚠️  IMPORTANTE:
echo    • No commits las credenciales (%CREDENTIALS_FILE%) al repositorio
echo    • Considera usar Workload Identity para mayor seguridad en CI/CD
echo    • El bucket tiene versionado habilitado para recuperar estados anteriores

pause