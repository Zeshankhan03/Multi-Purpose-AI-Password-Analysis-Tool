@echo off
echo Building MPAIPAT executable...
pyinstaller --clean --win-private-assemblies MPAIPAT.spec
echo Build complete!
pause