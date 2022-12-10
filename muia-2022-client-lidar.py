#!/usr/bin/python3

# --------------------------------------------------------------------------

print('### Script:', __file__)

# --------------------------------------------------------------------------

import math
import sys
import time

# import cv2 as cv
import numpy as np
import sim
from matplotlib import pyplot as plt
from bresenham import bresenham

# --------------------------------------------------------------------------
class Map:
    # https://github.com/AtsushiSakai/PythonRobotics/blob/master/Mapping/lidar_to_grid_map/lidar_to_grid_map.py
    def __init__(self):
        # Los valores del mapa real son: X entre [-6,6], Y entre [-3,3]
        self.resolution = 0.005
        self.x_min = -6
        self.y_min = -3
        self.x_max = 6
        self.y_max = 3
        self.x_size = int((self.x_max) * 2 / self.resolution)
        self.y_size = int((self.y_max) * 2 / self.resolution)
        print(self.x_size)
        print(self.y_size)


        # Occupancy map a 0.5 cuando no sabemos nada
        self.occupancy_map = np.ones(shape=[self.x_size, self.y_size]) / 2

    def updateMap(self, x, y, x_r, y_r):
        # coordenadas x e y del robot en el mapa
        x_r = int(round((x_r - self.x_min) / self.resolution))
        y_r = int(round((y_r - self.y_min) / self.resolution))
            
        for i in range(len(x)):
            # x coordinate of the the occupied area
            ix = int(round((x[i] - self.x_min) / self.resolution))
            # y coordinate of the the occupied area
            iy = int(round((y[i] - self.y_min) / self.resolution))
            laser_beams = bresenham(x_r, y_r, ix, iy)       # line form the lidar to the occupied point
            for laser_beam in laser_beams:
                self.occupancy_map[laser_beam[0]][-(laser_beam[1] + 1)] = 0.0  # free area 0.0
            self.occupancy_map[ix][-(iy + 1)] = 1.0         # occupied area 1.0
            self.occupancy_map[ix + 1][-(iy + 1)] = 1.0     # extend the occupied area
            self.occupancy_map[ix][-iy] = 1.0               # extend the occupied area
            self.occupancy_map[ix + 1][-iy] = 1.0           # extend the occupied area

# --------------------------------------------------------------------------

def getRobotHandles(clientID):
    # Motor handles
    _,lmh = sim.simxGetObjectHandle(clientID, 'Pioneer_p3dx_leftMotor',
                                     sim.simx_opmode_blocking)
    _,rmh = sim.simxGetObjectHandle(clientID, 'Pioneer_p3dx_rightMotor',
                                     sim.simx_opmode_blocking)

    # Sonar handles
    str = 'Pioneer_p3dx_ultrasonicSensor%d'
    sonar = [0] * 16
    for i in range(16):
        _,h = sim.simxGetObjectHandle(clientID, str % (i+1),
                                       sim.simx_opmode_blocking)
        sonar[i] = h
        sim.simxReadProximitySensor(clientID, h, sim.simx_opmode_streaming)

    # Camera handles
    _,cam = sim.simxGetObjectHandle(clientID, 'Pioneer_p3dx_camera',
                                        sim.simx_opmode_oneshot_wait)
    sim.simxGetVisionSensorImage(clientID, cam, 0, sim.simx_opmode_streaming)
    sim.simxReadVisionSensor(clientID, cam, sim.simx_opmode_streaming)

    _,robot = sim.simxGetObjectHandle(clientID, 'Pioneer_p3dx', sim.simx_opmode_blocking)
    sim.simxGetObjectPosition(clientID, robot, -1, sim.simx_opmode_streaming)
    sim.simxGetObjectOrientation(clientID, robot, -1, sim.simx_opmode_streaming)

    sim.simxGetStringSignal(clientID, 'Pioneer_p3dx_lidar_data', sim.simx_opmode_streaming)

    return [lmh, rmh], sonar, cam, robot

# --------------------------------------------------------------------------

def setSpeed(clientID, hRobot, lspeed, rspeed):
    sim.simxSetJointTargetVelocity(clientID, hRobot[0][0], lspeed,
                                    sim.simx_opmode_oneshot)
    sim.simxSetJointTargetVelocity(clientID, hRobot[0][1], rspeed,
                                    sim.simx_opmode_oneshot)

# --------------------------------------------------------------------------

def getSonar(clientID, hRobot):
    r = [1.0] * 16
    for i in range(16):
        handle = hRobot[1][i]
        e,s,p,_,_ = sim.simxReadProximitySensor(clientID, handle,
                                                 sim.simx_opmode_buffer)
        if e == sim.simx_return_ok and s:
            r[i] = math.sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2])

    return r

# --------------------------------------------------------------------------

# def getImage(clientID, hRobot):
#     img = []
#     err,r,i = sim.simxGetVisionSensorImage(clientID, hRobot[2], 0,
#                                             sim.simx_opmode_buffer)

#     if err == sim.simx_return_ok:
#         img = np.array(i, dtype=np.uint8)
#         img.resize([r[1],r[0],3])
#         img = np.flipud(img)
#         img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

#     return err, img

# --------------------------------------------------------------------------

def getImageBlob(clientID, hRobot):
    rc,ds,pk = sim.simxReadVisionSensor(clientID, hRobot[2],
                                         sim.simx_opmode_buffer)
    blobs = 0
    coord = []
    if rc == sim.simx_return_ok and pk[1][0]:
        blobs = int(pk[1][0])
        offset = int(pk[1][1])
        for i in range(blobs):
            coord.append(pk[1][4+offset*i])
            coord.append(pk[1][5+offset*i])

    return blobs, coord

# --------------------------------------------------------------------------

def avoid(sonar):
    if (sonar[3] < 0.25) or (sonar[4] < 0.25):
        lspeed, rspeed = +0.15, -0.65
    elif (sonar[2] < 0.3):
        lspeed, rspeed = +1.0, +0.6
    elif (sonar[5] < 0.3):
        lspeed, rspeed = +0.6, +1.0
    elif (sonar[1] < 0.42):
        lspeed, rspeed = +1.25, +0.85
    elif (sonar[6] < 0.42):
        lspeed, rspeed = +0.85, +1.25
    else:
        lspeed, rspeed = +1.5, +1.5

    return lspeed, rspeed

# --------------------------------------------------------------------------

def getRobotPose(clientID, hRobot):
    _,p = sim.simxGetObjectPosition(clientID, hRobot[3], -1, sim.simx_opmode_buffer)
    _,o = sim.simxGetObjectOrientation(clientID, hRobot[3], -1, sim.simx_opmode_buffer)
    return p[0], p[1], o[2]

# --------------------------------------------------------------------------

def getLidar(clientID, hRobot):
    _,data = sim.simxGetStringSignal(clientID, 'Pioneer_p3dx_lidar_data', sim.simx_opmode_buffer)
    return sim.simxUnpackFloats(data)

# --------------------------------------------------------------------------
def clean_data(data):
    """
    Reading LIDAR laser beams
    """

    # Create numpy array
    data = np.array(data)

    # Separate each point lecture
    if(len(data) < 3):
        return None
    return np.split(data, len(data)/3)


# --------------------------------------------------------------------------


def main():
    print('### Program started')

    print('### Number of arguments:', len(sys.argv), 'arguments.')
    print('### Argument List:', str(sys.argv))

    sim.simxFinish(-1) # just in case, close all opened connections

    #port = int(sys.argv[1])
    port = 19999
    clientID = sim.simxStart('127.0.0.1', port, True, True, 2000, 5)

    if clientID == -1:
        print('### Failed connecting to remote API server')

    else:
        print('### Connected to remote API server')
        hRobot = getRobotHandles(clientID)

        x_total = []
        y_total = []
        x_r = []
        y_r = []
        my_map = Map()

        # Real time plotting sacado de:
        # https://gist.github.com/superjax/33151f018407244cb61402e094099c1d
        '''
        plt.ion() # enable real-time plotting
        plt.figure(1) # create a plot
        '''

        while sim.simxGetConnectionId(clientID) != -1:

            x = []
            y = []

            # Perception
            sonar = getSonar(clientID, hRobot)
            # print '### s', sonar

            x_robot, y_robot, theta = getRobotPose(clientID, hRobot)
            # print('### pos', x, y, theta)

            data = getLidar(clientID, hRobot)
            data = clean_data(data)
            #print(data)
            #print(theta)

            ang = theta
            #print(ang)
            
            ## Transformo los puntos en las coordenadas sin rotación
            if data is None:
                pass
            else:                
                for point in data:
                    # Coordenadas en el eje sin rotación
                    p = np.array([point[0], point[1]])
                    p_1 = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]) @ p.T
                    real_point = np.array([p_1[0] + x_robot, p_1[1] + y_robot])
                    x.append(real_point[0])
                    y.append(real_point[1])
                    x_r.append(x_robot)
                    y_r.append(y_robot)
            

            '''
            ## Transformo los puntos en las coordenadas sin rotación
            if data is None:
                pass
            else:                
                for point in data:
                    # Coordenadas en el eje sin rotación
                    p = np.array([point[0], point[1]])
                    p_1 = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]) @ p.T
                    real_point = np.array([p_1[0] + x_robot, p_1[1] + y_robot])
                    x.append(real_point[0])
                    x_total.append(real_point[0])
                    y.append(real_point[1])
                    y_total.append(real_point[1])
                    x_r.append(x_robot)
                    y_r.append(y_robot)

            my_map.updateMap(x, y, x_robot, y_robot)

            plt.clf()       # Borrar la figura
            plt.imshow(my_map.occupancy_map.T, cmap="binary")
            plt.pause(0.005)
            '''

            # blobs, coord = getImageBlob(clientID, hRobot)
            # print('###  ', blobs, coord)

            # Planning
            lspeed, rspeed = avoid(sonar)

            # Action
            setSpeed(clientID, hRobot, lspeed, rspeed)
            time.sleep(0.01)
            #print('--------------------------------------------\n\n\n')

        # Convert points to numpy array
        x = np.array(x)
        y = np.array(y)
        x_r = np.array(x_r)
        y_r = np.array(y_r)

        # Plot the points
        plt.scatter(x,y)
        plt.scatter(x_r,y_r, c='magenta')
        plt.show()
        
        '''
        plt.ioff()
        plt.clf()
        plt.imshow(my_map.occupancy_map.T, cmap="binary")
        plt.show()
        '''

        print('### Finishing...')
        sim.simxFinish(clientID)

    print('### Program ended')

# --------------------------------------------------------------------------

if __name__ == '__main__':
    main()
