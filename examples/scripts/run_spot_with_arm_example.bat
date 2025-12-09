@echo off
REM Convenience launcher for a Spot-with-arm example task.
REM Adjust ISAACSIM_PATH if Isaac Sim is installed elsewhere.

set ISAACSIM_PATH=D:\isaacsim\isaac-sim-standalone-5.1.0-windows-x86_64

%ISAACSIM_PATH%\python.bat spot/examples/src/tasks/test_spot_with_arm_nav.py %*

