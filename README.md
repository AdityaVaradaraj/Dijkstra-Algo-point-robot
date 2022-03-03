# Dijkstra-Algo-point-robot

Demo Video: https://drive.google.com/file/d/1GMOK0OVZz0sz_2mIukr0CpLJXlxP2Pxp/view?usp=sharing

I have used dictionary for the open list and numpy array for closed list. The red represents obstacle space, purple circle is start location, black circle is goal location, green pixels are explored nodes and blue path is the computed optimal path from start to goal 

Libraries used:

1) cv2 (OpenCV) 
(Preferably v4.1.0. But since I have used only basic functions like cv2.circle, cv2.fillPoly and cv2.VideoWriter, I hope any version should work fine)

2) numpy

3) math

Instructions:
1) Unzip the zip file into one folder.

2) Open the terminal and type python3 Dijkstra-pathplanning-Aditya-Varadaraj.py

3) It should ask you for start x. Once you enter start x coordinate and press Enter, you will be prompted to enter start y and then similarly goal x and goal y

4) Once you have entered these coordinates, press Enter. If any of the coordinates are in obstacle space, it should prompt you to re-enter with appropriate message indicating which out of start and goal was in obstacle space. 

5) Once you have successfully entered valid start and goal coordinates, it should start the computation. Note that computation might take 5-15 mins for small-to-medium distances. But might take around 38 mins for large distances, i.e., far away start and goal locations.

6) Once computation is finished. It will print on the terminal "Goal reached". Then, wait for a minute or so. It should finish creating the video and should show the goal cost-to-come in the terminal and then the program should get terminated.

7) Now, in the folder where you ran the python file from should have a new video or overwritten video named "dijkstra_algo.mp4". Open and play the video to visualize the animation of the dijkstra algo for the start and goal values you had entered.

Notes: 

1) I have given clearance of 5mm from each point on the obstacle and boundary walls. The obstacles shown in the video in red color are inflated versions of original obstacles and hence include the 5mm clearance.  

2) Enter (x,y) in global coordinates, i.e., (0,0) being bottom left of the arena. The conversion from universal (x,y) to image (x,y) is done internally in the code in drawing and obstacle detection functions.
