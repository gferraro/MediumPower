# MediumPower
Dump of file generated from AI 


On Pi opencv from apt is at 4.6 so will need to make links from the opencv 4.6 version to opencv 2.4 for it work
sudo apt install libopencv-dev
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/lib/aarch64-linux-gnu/pkgconfig
sudo ln -s /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.6.0 /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.2

sudo ln -s /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.6.0 /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.22