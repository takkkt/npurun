@echo off
rem Chat with a GGUF model; ternary models (PQ2_0 / PTQ1_0) run on the AMD XDNA2 NPU + CPU.
rem   npurun-chat.cmd path\to\model.gguf [extra llama-cli args]
setlocal
set BIN=
for %%D in ("%~dp0." "%~dp0..\build\bin") do (
    if not defined BIN if exist "%%~D\llama-cli.exe" set "BIN=%%~D"
)
if not defined BIN (echo llama-cli.exe not found - build npurun first, see docs\guide.md & exit /b 1)
if "%~1"=="" (echo usage: %~nx0 model.gguf [llama-cli args] & exit /b 1)
"%BIN%\llama-cli.exe" -m "%~1" -t 6 --poll 100 --no-mmap -c 8192 %2 %3 %4 %5 %6 %7 %8 %9
endlocal
