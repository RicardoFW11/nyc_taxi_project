@echo off
title Pipeline NYC Taxi - Rebuild & Train
echo ========================================================
echo INICIANDO RECONSTRUCCIÓN Y ENTRENAMIENTO AUTOMATIZADO
echo ========================================================

:: 1. Limpieza total de contenedores y volumenes antiguos
echo [*] Paso 1/4: Limpiando contenedores previos...
docker-compose down -v

:: 2. Eliminación de imágenes para asegurar que lea los nuevos archivos (analytics.py, etc.)
echo [*] Paso 2/4: Eliminando imágenes antiguas...
docker rmi nyc_taxi_project-train_xgboost nyc_taxi_project-ui nyc_taxi_project-api -f

:: 3. Reconstrucción completa sin usar caché
echo [*] Paso 3/4: Reconstruyendo todo el ecosistema (esto tomará unos minutos)...
docker-compose build --no-cache

:: 4. Lanzar todo en segundo plano (Detached mode)
echo [*] Paso 4/4: Lanzando servicios en segundo plano...
docker-compose up -d

echo ========================================================
echo PROCESO INICIADO CORRECTAMENTE.
echo Puedes cerrar esta ventana. El entrenamiento seguirá en Docker.
echo ========================================================
pause