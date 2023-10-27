text
git clone <repository>
cd <repository>
cp -r ./gaze-estimation-demo <OPEN_MODEL_ZOO_DIR>/demos/
cd <OPEN_MODEL_ZOO_DIR>/demos
./build_demos.sh

cd <COMPILED_DIR>/ # For me, /home/name/omz_demos_build/intel64/Release/

cp -r <repository>/needs/* .
check your cameras on system via [camera_command]
open run.sh file via your favorite text editor, 
[sudo ./gaze_estimation_demo -d CPU -i <YOUR_CAM_ID> ...] modify it refer camera id on your system. note that your camera resolution would be 1280 * 720 pixels. 

All done. just enter ./run.sh .
