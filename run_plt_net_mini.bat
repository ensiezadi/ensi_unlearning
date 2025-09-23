@echo off
echo === PLT_NET_Mini Machine Unlearning Experiment ===
echo This script will run machine unlearning on PLT_NET_Mini dataset
echo.

REM Set parameters
set DATASET=PLTNetMini
set BACKBONE=RN50
set OUTPUT_DIR=results\plt_net_mini_experiment

echo Parameters:
echo - Dataset: %DATASET%
echo - Backbone: %BACKBONE%
echo - Output Directory: %OUTPUT_DIR%
echo.

echo Classes to forget (defined in forget_cls.py):
echo - Trifolium repens (White clover)
echo - Lactuca serriola (Prickly lettuce)
echo - Cirsium arvense (Creeping thistle)
echo.

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Run the experiment
echo Starting machine unlearning experiment...
python main.py --run_ds %DATASET% --backbone_arch %BACKBONE% --output_dir %OUTPUT_DIR% --seed 42

echo.
echo Experiment completed! Check results in: %OUTPUT_DIR%
pause