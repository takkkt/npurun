@echo off
rem OpenAI-compatible server at http://127.0.0.1:8080; ternary models run on the AMD XDNA2 NPU + CPU.
rem   npurun-server.cmd path\to\model.gguf [extra llama-server args]
setlocal
set BIN=
for %%D in ("%~dp0." "%~dp0..\build\bin") do (
    if not defined BIN if exist "%%~D\llama-server.exe" set "BIN=%%~D"
)
if not defined BIN (echo llama-server.exe not found - build npurun first, see docs\guide.md & exit /b 1)
if "%~1"=="" (echo usage: %~nx0 model.gguf [llama-server args] & exit /b 1)
"%BIN%\llama-server.exe" -m "%~1" -t 6 --poll 100 --no-mmap -c 8192 -np 1 --host 127.0.0.1 --port 8080 %2 %3 %4 %5 %6 %7 %8 %9
endlocal
