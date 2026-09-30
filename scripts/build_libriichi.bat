@echo off
REM Build libriichi Rust library
REM Requires: Rust toolchain, Visual Studio Build Tools, Python environment activated

echo Building libriichi...
cd /d "%~dp0.."

REM Activate conda environment if not already active
where maturin >nul 2>&1
if errorlevel 1 (
    echo ERROR: maturin not found. Please activate your conda environment first.
    echo   conda activate mortal
    exit /b 1
)

maturin develop --release --manifest-path libriichi\Cargo.toml
if errorlevel 1 (
    echo ERROR: Build failed.
    exit /b 1
)

echo Build successful!
echo You can verify by running: python -c "import libriichi; print('libriichi OK')"
