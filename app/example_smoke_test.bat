@echo off
REM Smoke test script for California Housing Prediction API (Windows version)
REM This script tests the API endpoints to ensure they're working correctly

setlocal enabledelayedexpansion

set API_URL=http://127.0.0.1:8000
set EXAMPLE_FILE=examples\example_request.json

echo 🧪 Starting smoke test for California Housing Prediction API...

REM Check if API is running
echo 📡 Checking if API is running at %API_URL%...
curl -s -f "%API_URL%/health" >nul 2>&1
if errorlevel 1 (
    echo ❌ API is not running at %API_URL%
    echo 💡 Please start the API first with: uvicorn app.main:app --reload
    echo 💡 Or run this script in the background: uvicorn app.main:app --reload ^&
    exit /b 1
)

echo ✅ API is running

REM Test health endpoint
echo �� Testing health endpoint...
curl -s "%API_URL%/health" > temp_health.json
set /p HEALTH_RESPONSE=<temp_health.json
echo Health response: !HEALTH_RESPONSE!

echo !HEALTH_RESPONSE! | findstr /C:"\"status\":\"ok\"" >nul
if errorlevel 1 (
    echo ❌ Health check failed
    del temp_health.json
    exit /b 1
) else (
    echo ✅ Health check passed
)

REM Test predict endpoint
echo 🔮 Testing predict endpoint...

REM Check if example file exists
if exist "%EXAMPLE_FILE%" (
    echo �� Using example file: %EXAMPLE_FILE%
    curl -s -X POST "%API_URL%/predict" -H "Content-Type: application/json" --data @"%EXAMPLE_FILE%" > temp_predict.json
) else (
    echo 📄 Example file not found, using inline JSON...
    REM Create inline JSON for testing
    set INLINE_JSON={"raw":{"MedInc":5.0,"HouseAge":20,"AveRooms":6.0,"AveBedrms":2.2,"Population":1500,"AveOccup":3.0,"Latitude":37.5,"Longitude":-121.9}}
    
    curl -s -X POST "%API_URL%/predict" -H "Content-Type: application/json" --data "!INLINE_JSON!" > temp_predict.json
)

set /p PREDICT_RESPONSE=<temp_predict.json
echo Prediction response: !PREDICT_RESPONSE!

REM Check if response contains predictions
echo !PREDICT_RESPONSE! | findstr /C:"\"predictions\"" >nul
if errorlevel 1 (
    echo ❌ Prediction endpoint failed - no 'predictions' key in response
    echo Response: !PREDICT_RESPONSE!
    del temp_health.json temp_predict.json
    exit /b 1
) else (
    echo ✅ Prediction endpoint working correctly
)

REM Test root endpoint
echo 🏠 Testing root endpoint...
curl -s "%API_URL%/" > temp_root.json
set /p ROOT_RESPONSE=<temp_root.json
echo Root response: !ROOT_RESPONSE!

echo !ROOT_RESPONSE! | findstr /C:"\"message\"" >nul
if errorlevel 1 (
    echo ❌ Root endpoint failed
    del temp_health.json temp_predict.json temp_root.json
    exit /b 1
) else (
    echo ✅ Root endpoint working correctly
)

REM Clean up temp files
del temp_health.json temp_predict.json temp_root.json

echo.
echo 🎉 All smoke tests passed!
echo ✅ Health endpoint: OK
echo ✅ Predict endpoint: OK
echo ✅ Root endpoint: OK
echo.
echo �� API is ready for use!
echo 📖 API documentation available at: %API_URL%/docs

endlocal
