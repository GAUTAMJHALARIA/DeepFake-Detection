@echo off
echo MP4 Batch Converter for Deepfake Detection System
echo ================================================
echo.
echo This script converts all MP4 files in the current folder to browser-compatible H.264 format.
echo.
echo Requirements:
echo - FFmpeg must be installed and in your PATH
echo - Run this script in the folder containing your MP4 files
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul
echo.
echo Starting conversion...
echo.

set count=0
for %%f in (*.mp4) do (
    set /a count+=1
    echo Converting %%f...
    ffmpeg -i "%%f" -c:v libx264 -profile:v baseline -c:a aac -y "converted_%%f"
    if !errorlevel! equ 0 (
        echo ✓ Successfully converted %%f
    ) else (
        echo ✗ Failed to convert %%f
    )
    echo.
)

echo.
echo Conversion complete!
echo Converted %count% files.
echo.
echo Original files are preserved. Converted files have "converted_" prefix.
echo.
pause
