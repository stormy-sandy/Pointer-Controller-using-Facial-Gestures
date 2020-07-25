cd C:\Program Files (x86)\IntelSWTools\openvino_2020.3.194\bin
setupvars.bat
cd C:\Users\dell\Documents\GitHub\Pointer-Controller-using-Facial-Gestures
python main.py -fdm intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001  -fldm intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 -hpem intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -gem intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -i bin/demo.mp4 --device "CPU" -m_speed "medium" -m_prec "low" -flags fd hp fld ge