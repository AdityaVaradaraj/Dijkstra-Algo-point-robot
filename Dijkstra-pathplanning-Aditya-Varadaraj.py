#!/usr/bin/env python
import numpy as np
import math
import cv2

xlim = 400
ylim = 250

class Dijkstra:
    def __init__(self, start_state, goal_state):
        self.n_i = 1 # Node Index counter
        # Node Format: (c2c cost, node index, parent index, start state)
        self.start_state = start_state # Start node 
        self.goal_state = goal_state # Goal (x,y)
        self.goal_cost = None
        self.open = {self.start_state : (0, self.n_i, 0)}
        self.map = np.zeros([ylim+1, xlim+1, 3], dtype=np.uint8)
        self.map.fill(255)
        
        self.closed = None
        self.path = np.array([[]], dtype=object)
        self.exit = False
        
    #---------- Drawing the obstacle and free space and saving visualization --------
    def DrawMap(self):
        th_5 = math.atan2((65-40), (36-115))
        th_6 = math.atan2((70-40), (80-115))
        th_7 = math.atan2((150-70), (105-80))
        th_8 = math.atan2((150-65), (105-36))
        obs_1 = np.array([\
            [36 + 5*np.cos(np.pi/2+th_8), ylim-185 + 5*np.sin(np.pi/2+th_8)], [36, ylim-185], [36 + 5*np.cos(np.pi/2+th_5), ylim-185 + 5*np.sin(np.pi/2+th_5)],\
            [115 + 5*np.cos(np.pi/2+ th_5), 40 + 5*np.sin(np.pi/2 + th_5)], [115, 40], [115 - 5*np.cos(np.pi/2+ th_6), 40 - 5*np.sin(np.pi/2 + th_6)],\
            [80 - 5*np.cos(np.pi/2 + th_6), ylim-180 - 5*np.sin(np.pi/2 + th_6)], [80, ylim-180], [ 80 - 5*np.cos(np.pi/2 + th_7), ylim-180 - 5*np.sin(np.pi/2 + th_7)],\
            [105 - 5*np.cos(np.pi/2 + th_7), ylim-100 - 5*np.sin(np.pi/2+ th_7)], [105, ylim-100], [105 + 5*np.cos(np.pi/2 + th_8), ylim-100 + 5*np.sin(np.pi/2+ th_8)]\
            ], np.int32)
        
        obs_1= obs_1.reshape((-1,1,2))
        cv2.fillPoly(self.map, [obs_1], (0,0,255))
        obs_1_act = np.array([[36, ylim-185], [115, 40],[ 80, ylim-180],[105, ylim-100]], np.int32)
        for obs in obs_1_act:
            self.map = cv2.circle(self.map, (obs[0], obs[1]), 5, (0,0,255), thickness = -1)
        obs_2 = np.array([[200 + 5*np.cos(np.pi/2 + np.pi - np.pi/6), 109.59 + 5*np.sin(np.pi/2 + np.pi - np.pi/6)],[200, int(109.59)],[200 + 5*np.cos(np.pi/6- np.pi/2), 109.59 + 5*np.sin(np.pi/6- np.pi/2)], [235 + 5*np.cos(np.pi/6- np.pi/2), 129.8 + 5 *np.sin(np.pi/6-np.pi/2)], [235, int(129.8)], [235 + 5, 129.8], [235+5, int(170.20)], [235, int(170.20)], [235 + 5*np.cos(np.pi/2 - np.pi/6), 170.20 + 5*np.sin(np.pi/2 - np.pi/6)], [200 + 5*np.cos(np.pi/2-np.pi/6), 190.41 + 5*np.sin(np.pi/2 - np.pi/6)],[200, int(190.41)], [200 + 5*np.cos(np.pi/2 + np.pi/6), 190.41 + 5*np.sin(np.pi/2+np.pi/6)],[165 + 5*np.cos(np.pi/2+np.pi/6), 170.20 + 5*np.sin(np.pi/2 + np.pi/6)],[165, int(170.20)], [165-5, int(170.20)], [165-5, int(129.8)],[165, int(129.8)], [165 + 5*np.cos(np.pi/2 + np.pi - np.pi/6), 129.8 + 5*np.sin(np.pi/2 + np.pi - np.pi/6)]], np.int32)
        obs_2= obs_2.reshape((-1,1,2))
        cv2.fillPoly(self.map, [obs_2], (0,0,255))
        obs_2_act = np.array([[200, int(109.59)],[235, int(129.8)],[235, int(170.20)],[200, int(190.41)],[165, int(170.20)],[165, int(129.8)]], np.int32)
        for obs in obs_2_act:
            self.map = cv2.circle(self.map, (obs[0], obs[1]), 5, (0,0,255), thickness = -1)
        self.map = cv2.circle(self.map, (300, 65), 40 + 5, (0,0,255), thickness=-1)
        
        self.map = cv2.circle(self.map, (int(self.start_state[0]), int(ylim - self.start_state[1])), 3, (255,0,255), thickness = -1)
        self.map = cv2.circle(self.map, (int(self.goal_state[0]), int(ylim - self.goal_state[1])), 3, (0,0,0), thickness = -1) 
        bndy_1 = np.array([[0,0],[0,5],[400,5],[400,0]])
        bndy_2 = np.array([[0,0],[5,0],[5,250],[0,250]])
        bndy_3 = np.array([[0,250],[0,245],[400,245],[400,250]])
        bndy_4 = np.array([[400,0],[395,0],[395,250],[400,250]])
        cv2.fillPoly(self.map, [bndy_1], (0,0,255))
        cv2.fillPoly(self.map, [bndy_2], (0,0,255))
        cv2.fillPoly(self.map, [bndy_3], (0,0,255))
        cv2.fillPoly(self.map, [bndy_4], (0,0,255))

        frameSize = (400, 250)
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        out  = cv2.VideoWriter('dijkstra_algo.mp4', fourcc, 250, frameSize)
        out.write(self.map)
        for i in range(len(self.closed)):
            x = self.closed[i][3][0]
            y = ylim - self.closed[i][3][1]
            self.map[y][x] = [0,255,0]
            out.write(self.map)
        for i in range(len(self.path)):
            x = self.path[i][0]
            y = ylim - self.path[i][1]
            self.map[y][x] = [255,0,0]
            out.write(self.map)
        for i in range(1250):
            out.write(self.map)
        out.release()
        print('Cost To Come for Goal: ' + str(self.goal_cost))
        

    # -------------- Checking Whether point is in Obstacle space --------------
    # X is (x,y) in Universal frame
    # Within function it is transformed into image frame and
    # then using half-plane equations of obstacles in image frame
    # checking is done
    def isObstacle(self,X):
        th_5 = math.atan2((65-40), (36-115))
        th_6 = math.atan2((70-40), (80-115))
        th_7 = math.atan2((150-70), (105-80))
        th_8 = math.atan2((150-65), (105-36))
        x = X[0]
        y = 250 - X[1]
        m1 = np.tan(-np.pi/6)
        fh1 = y - m1*x + m1*200 - 109.59
        fh3 = y - m1*x + m1*200 - 190.41
        m3 = np.tan(np.pi/6)
        fh2 = y - m3*x + m3*200 - 109.59
        fh4 = y - m3*x + m3*200 - 190.41
        
        m5 = (65-40)/(36-115)
        m6 = (70-40)/(80-115)
        m7 = (150-70)/(105-80)
        m8 = (150-65)/(105-36)
        fp1 = y - m5*x + m5*115 - 40
        fp2 = y - m6*x + m6*115 - 40
        fp3 = y - m7*x + m7*105 - 150
        fp4 = y - m8*x + m8*105 - 150
        
        fh11 = y - (-1/m1)*x + (-1/m1)*165 - 129.8
        fh12 = y - (-1/m1)*x + (-1/m1)*200 - 109.59
        fh13 = y - m1*x + m1*(200 + 5*np.cos(np.pi/2 + np.pi - np.pi/6)) - (109.59 + 5*np.sin(np.pi/2 + np.pi - np.pi/6))
        fh21 = y - (-1/m3)*x + (-1/m3)*200 - 109.59
        fh22 = y - (-1/m3)*x + (-1/m3)*235 - 129.8
        fh23 = y - m3*x + m3*(200 + 5*np.cos(np.pi/6- np.pi/2)) - (109.59 + 5*np.sin(np.pi/6- np.pi/2))
        fh31 = y - (-1/m1)*x + (-1/m1)*235 - 170.20
        fh32 = y - (-1/m1)*x + (-1/m1)*200 - 190.41
        fh33 = y - m1*x + m1*(235 + 5*np.cos(np.pi/2 - np.pi/6)) - (170.20 + 5*np.sin(np.pi/2 - np.pi/6))
        fh41 = y - (-1/m3)*x + (-1/m3)*200 - 190.41
        fh42 = y - (-1/m3)*x + (-1/m3)*165 - 170.20
        fh43 = y - m3*x + m3*(200 + 5*np.cos(np.pi/2 + np.pi/6)) - (190.41 + 5*np.sin(np.pi/2 + np.pi/6))

        fp11 =  y - (-1/m5)*x + (-1/m5)*36 - 65
        fp12 =  y - (-1/m5)*x + (-1/m5)*115 - 40
        fp13 =  y - m5*x + m5*(36 + 5*np.cos(np.pi/2 + th_5)) - (65 + 5*np.sin(np.pi/2 + th_5))
        fp21 =  y - (-1/m6)*x + (-1/m6)*115 - 40
        fp22 =  y - (-1/m6)*x + (-1/m6)*80 - 70
        fp23 =  y - m6*x + m6*(115 - 5*np.cos(np.pi/2 + th_6)) - (40 - 5*np.sin(np.pi/2 + th_6))
        fp31 =  y - (-1/m7)*x + (-1/m7)*80 - 70
        fp32 =  y - (-1/m7)*x + (-1/m7)*105 - 150
        fp33 =  y - m7*x + m7*(80 - 5*np.cos(np.pi/2 + th_7)) - (70 - 5*np.sin(np.pi/2 + th_7))
        fp41 =  y - (-1/m8)*x + (-1/m8)*105 - 150
        fp42 =  y - (-1/m8)*x + (-1/m8)*36 - 65
        fp43 =  y - m8*x + m8*(105 + 5*np.cos(np.pi/2 + th_8)) - (150 + 5*np.sin(np.pi/2 + th_8))

        is_Obs2_padding = \
            (x>=160 and x<=165 and y>=129.8 and y<=170.20) or (x>=235 and x<=240 and y>=129.8 and y<=170.20) or\
            (fh1 <= 0 and fh13 >=0 and fh11 <=0 and fh12 >=0) or (fh12 <=0 and fh21 <=0 and pow((x-200), 2) + pow((y-109.59), 2) <= 25) or\
            (fh2 <=0 and fh23 >=0 and fh21 >=0 and fh22 <=0) or (fh22 >=0 and y <=129.8 and pow((x-235), 2) + pow((y-129.8), 2) <= 25) or\
            (y >= 170.20 and fh31 <=0 and pow((x-235), 2) + pow((y-170.20), 2) <= 25) or (fh31 >=0 and fh3 >=0 and fh33 <=0 and fh32 <=0) or\
            (fh32 >= 0 and fh41 >=0 and pow((x-200), 2) + pow((y-190.41), 2) <= 25) or (fh41 <=0 and fh42 >=0 and fh4 >=0 and fh43 <=0) or\
            (fh42 <= 0 and y >= 170.20 and pow((x-165), 2) + pow((y-170.20), 2) <= 25) or (y <= 129.8 and fh11 >=0 and pow((x-165), 2) + pow((y-129.8), 2) <= 25)

        is_Obs1_padding = \
            (fp1 <=0 and fp13 >=0 and fp11 <=0 and fp12 >=0) or (fp12<=0 and fp21 <=0 and pow((x-115),2) + pow((y-40),2) <= 25) or\
            (fp21 >= 0 and fp22 <= 0 and fp2 >= 0 and fp23 <=0) or (fp2 >=0 and fp3 <=0 and pow((x-80),2) + pow((y-70),2) <= 25) or\
            (fp3 <=0 and fp33 >=0 and fp31 >=0 and fp32 <=0) or (fp32>=0 and fp41 >=0 and pow((x-105),2) + pow((y-150),2) <= 25) or\
            (fp41 <=0 and fp42 >=0 and fp4 >=0 and fp43 <=0) or (fp42<=0 and fp11 >=0 and pow((x-36),2) + pow((y-65),2) <= 25)
        
        is_Obs3_with_padding = (pow((x-300),2) + pow((y-65),2)) <= pow(45,2)
        
        is_Obs2 = (x >= 165 and x <= 235 and fh1 >= 0 and fh3 <= 0 and fh2 >=0 and fh4 <= 0)
        
        is_Obs1 = (fp1 <=0 and fp2 >= 0 and fp3 <= 0 and fp4 >= 0)
        
        is_Bndy_padded = (x>=0 and x<=5) or (x>=395 and x<=400) or (y>=0 and y<=5) or (y>=245 and y<=250)

        return (is_Obs1 or is_Obs2 or is_Obs3_with_padding or is_Obs1_padding or is_Obs2_padding or is_Bndy_padded)
    

    # ---------------- Action Functions (8-ActionSet) -------------
    def ActionMoveLeft(self, current_node_key):
        X = np.copy(list(current_node_key))
        if X[0] != 0:
            X[0] -= 1
            success = True
            return success, tuple(X)
        else:
            success = False
            return success, self.start_state

    def ActionMoveUp(self, current_node_key):
        X = np.copy(list(current_node_key))
        if X[1] != ylim:
            X[1] += 1
            success = True
            return success, tuple(X)
        else:
            success = False
            return success, self.start_state

    def ActionMoveRight(self, current_node_key):
        X = np.copy(list(current_node_key))
        if X[0] != xlim:
            X[0] += 1
            success = True
            return success, tuple(X)
        else:
            success = False
            return success, self.start_state

    def ActionMoveDown(self, current_node_key):
        X = np.copy(list(current_node_key))
        if X[1] != 0:
            X[1] -= 1
            success = True
            return success, tuple(X)
        else:
            success = False
            return success, self.start_state

    def ActionMoveLeftUp(self, current_node_key):
        X = np.copy(list(current_node_key))
        if X[0] != 0 and X[1] != ylim:
            X[0] -= 1
            X[1] += 1
            success = True
            return success, tuple(X)
        else:
            success = False
            return success, self.start_state

    def ActionMoveRightUp(self, current_node_key):
        X = np.copy(list(current_node_key))
        if X[0] != xlim and X[1] != ylim:
            X[0] += 1
            X[1] += 1
            success = True
            return success, tuple(X)
        else:
            success = False
            return success, self.start_state

    def ActionMoveRightDown(self, current_node_key):
        X = np.copy(list(current_node_key))
        if X[0] != xlim and X[1] != 0:
            X[0] += 1
            X[1] -= 1
            success = True
            return success, tuple(X)
        else:
            success = False
            return success, self.start_state

    def ActionMoveLeftDown(self, current_node_key):
        X = np.copy(list(current_node_key))
        if X[0] != 0 and X[1] != 0:
            X[0] -= 1
            X[1] -= 1
            success = True
            return success, tuple(X)
        else:
            success = False
            return success, self.start_state

    # ----------------- Backtracking ----------------
    def backtrack(self,g):
        c = np.copy(g)
        c_i = g[1]
        indices = self.closed[:, 1]
        crd_list = self.closed[:, 3]
        while c_i != 0:
            self.path = np.append(self.path, crd_list[np.where(indices == c_i)[0]])
            c_i = c[2]
            if c_i == 0:
                break
            c = self.closed[np.where(indices == c_i)[0][0], :]
            
        self.path = np.flip(self.path, axis=0)

    # ------------------ Dijkstra Algo -------------------
    def dijkstra(self):
        j = 0 # iteration count
        while (self.open) and (not self.exit):
            # Popping from dictionary like a minimum priority queue
            n_k =  min(self.open.items(), key=lambda x: x[1][0])[0]
            n_v = self.open[n_k]
            del self.open[n_k]

            if j==0:
                # Start node
                self.closed = np.array([[n_v[0], n_v[1], n_v[2], n_k]], dtype=object) 
            else:
                self.closed = np.append(self.closed, [np.array([n_v[0], n_v[1], n_v[2], n_k], dtype=object)], axis=0)
            
            # If goal, do exit loop and do backtracking and then visualization
            if n_k == self.goal_state:
                self.goal_cost = n_v[0]
                self.exit = True
                print('Goal reached')
                break
            
            # Get resultant (x,y) and possibility of each action in actionset
            L_exists, L_Node = self.ActionMoveLeft(n_k)
            U_exists, U_Node = self.ActionMoveUp(n_k)
            R_exists, R_Node = self.ActionMoveRight(n_k)
            D_exists, D_Node = self.ActionMoveDown(n_k)
            LU_exists, LU_Node = self.ActionMoveLeftUp(n_k)
            RU_exists, RU_Node = self.ActionMoveRightUp(n_k)
            RD_exists, RD_Node = self.ActionMoveRightDown(n_k)
            LD_exists, LD_Node = self.ActionMoveLeftDown(n_k)
            
            # 1) If node exists and not in obstacle space
            # 2) If node is not in closed list
            # 3) If node is not in Open list 
            #        --> Calculate C2C and parent index and add to open dictionary
            # 4) If node is already in Open list 
            #        --> Based on old and new C2C, decide whether to update C2C and parent 
            if L_exists and (not self.isObstacle(L_Node)):
                in_closed = [n == L_Node for n in self.closed[:,3]]
                if np.count_nonzero(in_closed) ==0:
                    c2c = n_v[0] + 1
                    p_i = n_v[1]
                    if not (L_Node in self.open.keys()):
                        self.n_i += 1
                        self.open[L_Node] = (c2c, self.n_i, p_i)
                    else:
                        if(self.open[L_Node][0] > c2c):
                            self.open[L_Node] = (c2c, self.open[L_Node][1], p_i)
                        
            if U_exists and (not self.isObstacle(U_Node)):
                in_closed = [n == U_Node for n in self.closed[:,3]]
                if np.count_nonzero(in_closed) == 0:
                    c2c = n_v[0] + 1
                    p_i = n_v[1]
                    if not (U_Node in self.open.keys()):
                        self.n_i += 1
                        self.open[U_Node] = (c2c, self.n_i, p_i)
                    else:
                        if(self.open[U_Node][0] > c2c):
                            self.open[U_Node] = (c2c, self.open[U_Node][1], p_i)
            if R_exists and (not self.isObstacle(R_Node)):
                in_closed = [n == R_Node for n in self.closed[:,3]]
                if np.count_nonzero(in_closed) == 0:
                    c2c = n_v[0] + 1
                    p_i = n_v[1]
                    if not (R_Node in self.open.keys()):
                        self.n_i += 1
                        self.open[R_Node] = (c2c, self.n_i, p_i)
                    else:
                        if(self.open[R_Node][0] > c2c):
                            self.open[R_Node] = (c2c, self.open[R_Node][1], p_i)
            if D_exists and (not self.isObstacle(D_Node)):
                in_closed = [n == D_Node for n in self.closed[:,3]]
                if np.count_nonzero(in_closed) == 0:
                    c2c = n_v[0] + 1
                    p_i = n_v[1]
                    if not (D_Node in self.open.keys()):
                        self.n_i += 1
                        self.open[D_Node] = (c2c, self.n_i, p_i)
                    else:
                        if(self.open[D_Node][0] > c2c):
                            self.open[D_Node] = (c2c, self.open[D_Node][1], p_i)
            
            if LU_exists and (not self.isObstacle(LU_Node)):
                in_closed = [n == LU_Node for n in self.closed[:,3]]
                if np.count_nonzero(in_closed) == 0:
                    c2c = n_v[0] + 1.4
                    p_i = n_v[1]
                    if not (LU_Node in self.open.keys()):
                        self.n_i += 1
                        self.open[LU_Node] = (c2c, self.n_i, p_i)
                    else:
                        if(self.open[LU_Node][0] > c2c):
                            self.open[LU_Node] = (c2c, self.open[LU_Node][1], p_i)

            if RU_exists and (not self.isObstacle(RU_Node)):
                in_closed = [n == RU_Node for n in self.closed[:,3]]
                if np.count_nonzero(in_closed) == 0:
                    c2c = n_v[0] + 1.4
                    p_i = n_v[1]
                    if not (RU_Node in self.open.keys()):
                        self.n_i += 1
                        self.open[RU_Node] = (c2c, self.n_i, p_i)
                    else:
                        if(self.open[RU_Node][0] > c2c):
                            self.open[RU_Node] = (c2c, self.open[RU_Node][1], p_i)
            
            if RD_exists and (not self.isObstacle(RD_Node)):
                in_closed = [n == RD_Node for n in self.closed[:,3]]
                if np.count_nonzero(in_closed) == 0:
                    c2c = n_v[0] + 1.4
                    p_i = n_v[1]
                    if not (RD_Node in self.open.keys()):
                        self.n_i += 1
                        self.open[RD_Node] = (c2c, self.n_i, p_i)
                    else:
                        if(self.open[RD_Node][0] > c2c):
                            self.open[RD_Node] = (c2c, self.open[RD_Node][1], p_i)
            
            if LD_exists and (not self.isObstacle(LD_Node)):
                in_closed = [n == LD_Node for n in self.closed[:,3]]
                if np.count_nonzero(in_closed) == 0:
                    c2c = n_v[0] + 1.4
                    p_i = n_v[1]
                    if not (LD_Node in self.open.keys()):
                        self.n_i += 1
                        self.open[LD_Node] = (c2c, self.n_i, p_i)
                    else:
                        if(self.open[LD_Node][0] > c2c):
                            self.open[LD_Node] = (c2c, self.open[LD_Node][1], p_i)
            j += 1
        
        # If path to goal is found --> backtrack and save visualization in mp4 file
        # Else --> No solution found 
        if self.exit:
            self.backtrack(self.closed[-1, :])
            self.DrawMap()
        else:
            print('No solution found')
            if not self.open:
                print('Empty Open')


if __name__ == '__main__':
    start = [0,0]
    goal = [400,250]
    # Keep Taking input start state and goal state from user 
    # until he gives valid inputs (valid meaning not in obstacle space)
    
    start[0] = int(input('Enter start x: '))
    start[1] = int(input('Enter start y: '))
    goal[0] = int(input('Enter goal x: '))
    goal[1] = int(input('Enter goal y: '))
    start = tuple(start)
    goal = tuple(goal)
    D = Dijkstra(start,goal)
    while(D.isObstacle(start) or D.isObstacle(goal)):
        if D.isObstacle(start):
            print('Start state is in obstacle')
        if D.isObstacle(goal):
            print('Goal state is in obstacle')
        print('Re-enter start and goal states: ')
        start = [0,0]
        goal = [400,250]
        start[0] = int(input('Enter start x: '))
        start[1] = int(input('Enter start y: '))
        goal[0] = int(input('Enter goal x: '))
        goal[1] = int(input('Enter goal y: '))
        start = tuple(start)
        goal = tuple(goal)
        D = Dijkstra(start,goal)
    
    D.dijkstra()
    
