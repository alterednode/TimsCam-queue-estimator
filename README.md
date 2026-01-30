# TimsCam-queue-estimator

Permission from UBCO must be obtained if you wish to host this project if you are using it for anything beyond personal use, according to the Terms of use and Copyright information linked on the [Current Students](https://ok.ubc.ca/current-students) page

If using this for a different m3u8 playlist / stream, obtain permission from whomever is hosting it if needed.

Created with python 3.14

It uses OpenVINO to optimize YOLOv8 for intel cpus, as I am planning to run this on an i7-6700 at home, and develop it on my framework with a 13th gen intel CPU

This uses the TimCam "playlist" and reads in images, proccesses them with YOLOv8 and attempts to determine
 - how many people are in line for the tim hortons (implemented)
 - how many people are in the courtyard (not implemented)

The current plan is to make this info accessable via a public API that I am planning to create on timcam-count.cheyne.dev
