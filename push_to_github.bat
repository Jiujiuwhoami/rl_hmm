@echo off
echo ====================================
echo 推送到 GitHub
echo ====================================
echo.

cd /d "%~dp0"

echo 正在推送到 GitHub...
echo 请输入您的 GitHub 用户名和密码/令牌
echo.

git push -u origin main

echo.
echo ====================================
echo 推送完成！
echo ====================================
pause
