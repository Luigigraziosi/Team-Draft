import matplotlib.pyplot as plt
import math
import numpy as np
import time as tm
import sys
import rospy

# Enum is a class in python for creating enumerations, which are a set of symbolic names (members) bound to unique, constant values
from enum import Enum
from expand_array import expand_array
from insert_open import insert_open
from field import mapgenerator
from min_fn import min_fn
from node_index import node_index
from visual_path import visual_path
from velocity_profile import velocity_profile

from std_msgs.msg import *       # Standard ROS messages
from geometry_msgs.msg import *  # Geometry messages
from mavros_msgs.msg import *    # MavROS messages
from mavros_msgs.srv import *    # MavROS messages for services

from tf.transformations import euler_from_quaternion, quaternion_from_euler

flag_to = False
flag_cl = False
pose = Pose()
done = False
show_animation = True

# Class of direct command of MavROS
class mavros_cmd:

    # Inizialized the class
    def __init__(self):
        self.landService = rospy.ServiceProxy('mavros/cmd/land', mavros_msgs.srv.CommandTOL)
        self.armService = rospy.ServiceProxy('mavros/cmd/arming', mavros_msgs.srv.CommandBool)
        self.flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
        self.takeoffService = rospy.ServiceProxy('mavros/cmd/takeoff', mavros_msgs.srv.CommandTOL)
        self.changeframeService = rospy.ServiceProxy('mavros/setpoint_velocity/mav_frame',mavros_msgs.srv.SetMavFrame)
        self.alt = 0.6

    # Arm the motors via traditional ROS service invokation
    def setArm(self):
        rospy.wait_for_service('mavros/cmd/arming')
        try:
            self.armService(True)
            print('Motors armed')
        except rospy.ServiceException:
            print ('Service arming call failed!!!')

    # Set the Flight mode to "GUIDED" to permit to control the flight via command
    def setGuidedMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            self.flightModeService(custom_mode ='GUIDED')
            print('Flight mode: Guided')
        except rospy.ServiceException:
            print ('Service set_mode call failed, Guided Mode could not be set!!!')

    # Call of Takeoff service, this command is customized to obtain a initial altitude
    # the altitude can be modified by waypoints (AVOID TO MODIFY alt!!!)
    def setTakeoff(self):
        rospy.wait_for_service('mavros/cmd/takeoff')
        try:
            self.takeoffService(altitude = self.alt)
            print('Takeoff')
        except rospy.ServiceException:
            print('Service takeoff call failed!!!')

    # Call of Landing service: it performs the landing and disarm the motor after it
    def setLand(self):
        rospy.wait_for_service('mavros/cmd/land')
        try:
            self.landService(altitude = 0.05)
            print('Landing')
        except rospy.ServiceException:
            print ('Service land call failed!!!')

    # Call of MaV Frame:  it change between the global and the local RF
    def setMAVFrame(self):
        rospy.wait_for_service('mavros/setpoint_velocity/mav_frame')
        try:
            self.changeframeService(mav_frame = 8)
            print('RF changed')
        except rospy.ServiceException:
            print('Change Frame failed!!!')


# This function is used to obtain a precise altitude takeoff CHECK
def pose_cb(msg):
    global flag_to
    global pose
    pose = msg.pose
    if msg.pose.position.z >= 0.5:
        flag_to = True


def altitude_cb(msg):
    global flag_cl
    distance = msg.data
    if distance > 1.0:
        flag_cl = True

def publish_time(publisher, message, seconds, rate):
        time_start = rospy.get_time()
        while (rospy.get_time() - time_start) < seconds:
            publisher.publish(message)
            rate.sleep()

def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")


def ast(start_dot, target_dot, dx, obst, x_max, y_max, z_max):

    # Gain for the selectivity of the cost function
    gain_cost = 1.5
    # Start location [x y]
    start = np.array([int(start_dot[0]), int(start_dot[1]), int(start_dot[2])])
    # Target location [x y]
    target = np.array([int(target_dot[0]), int(target_dot[1]), int(target_dot[2])])
    # Call the obstacle generation function

    ## Preliminary procedures

    #start_time_all = tm.time()

    # CLOSED contains all the nodes that don't have to be evaluated
    # CLOSED ==> X val | Y val | Z val
    # I've created the closed array and initialized it with the starting node because I don't know how to create an empty 1x2 array
    CLOSED = start
    # Put all obstacles on the Closed List
    i=0
    for i in range(0,np.shape(obst)[0],1):
        CLOSED = np.vstack([CLOSED, [obst[i, 0], obst[i, 1], obst[i, 2]]])
    # Update the dimension of the closed list whit the number of obstacles
    closed_count = i+1
    # Set the starting node as the firs node to expand
    xNode = start[0]
    yNode = start[1]
    zNode = start[2]
    # Assign a zero cost to the starting node (heuristic_function)
    path_cost = 0
    # Calculate starting distance to the target (cost_function)
    goal_distance = np.sqrt((target[0] - start[0])**2+(target[1] - start[1])**2+(target[2] - start[2])**2) * gain_cost
    # OPEN list stores all the coordinates of all the nodes that have to be expanded, the coordinates of the # parent node and the 3 functions linked to that node: heuristic (distance from the start), cost (distance to the target), heuristic + cost.
    # OPEN ==> IS ON LIST 1/0 |X val |Y val |Z val |Parent X val |Parent Y val |Parent Z val|h(n) |g(n)|f(n)|
    # Same as CLOSED
    OPEN = np.array([0,0,0,0,0,0,0,0,0,0])
    OPEN = np.vstack([OPEN,[0, xNode, yNode, zNode, xNode, yNode, zNode, path_cost, goal_distance, path_cost + goal_distance]])
    # Update the open list dimension with the first node added
    open_count = 2
    # Set the flag which tells us the impossibility to find a path
    NoPath = 0
    new_node = []

    # Main algorithm

    # Here there's the main loop which is iterated until the coordindates are
    # the same of the target node or is impossible to find a path
    while (xNode != target[0] or yNode!=target[1] or zNode!=target[2]) and NoPath == 0:
        # Expand the actual node exploring the adjacent nodes
        exp_node = expand_array(xNode, yNode, zNode, path_cost, target[0], target[1], target[2], CLOSED, x_max, y_max, z_max,gain_cost)
        # Counter which counts the number of allowed nodes created
        exp_count = np.shape(exp_node)[0]
        # In this loop we check if a node expanded is already present in the OPEN list and if the cost now calculated has decreased we update his cost.
        if np.any(exp_node):
            for i in range(0,exp_count,1):
                new_node = 1
                for j in range(1,open_count,1):
                    #Check if the node is already present in the open list
                    if (exp_node[i,0] == OPEN[j,1] and exp_node[i,1] == OPEN[j,2] and exp_node[i,2] == OPEN[j,3]):
                        # If yes it match the cost values for the same node and take the minimum value
                        OPEN[j,-1] = np.amin([OPEN[j,-1],exp_node[i,-1]])
                        new_node = 0
                        # If the cost has been updated (also if two costs are the same, the one already in OPEN e and the one found by the expansion then update all others parameter.
                        if OPEN[j, -1] == exp_node[i, -1]:
                            #UPDATE PARENTS,gn,hn
                            OPEN[j, 4] = xNode
                            OPEN[j, 5] = yNode
                            OPEN[j, 6] = zNode
                            OPEN[j, -3] = exp_node[i, -3]
                            OPEN[j, -2] = exp_node[i, -2]
                        #End of minimum fn check
                    #End of node check
                #End of j for
                # If the flag new_node is still 0 it means the node has never been explored before so it has to be added to the OPEN list
                if new_node == 1 :
                    OPEN = np.vstack([OPEN,[insert_open(exp_node[i,0],exp_node[i,1],exp_node[i,2],xNode,yNode,zNode,exp_node[i,-3],exp_node[i,-2],exp_node[i,-1])]])
                    open_count += 1
                #End of insert new element into the OPEN list
            #End of i for

        #Find out the node with the smallest fn
        index_min_node = min_fn(OPEN,open_count,target[0],target[1],target[2])
        # La funzione da -1 se non ci sono piu nodi da espandere
        if (index_min_node != -1):
            #Set xNode and yNode to the node with minimum fn
            xNode = int(OPEN[index_min_node,1])
            yNode = int(OPEN[index_min_node,2])
            zNode = int(OPEN[index_min_node,3])
            #Update the cost of reaching the parent node
            path_cost = OPEN[index_min_node,-3]
            #Move the Node to list CLOSED
            CLOSED = np.vstack([CLOSED,[xNode,yNode,zNode]])
            closed_count += 1
            # Instead of remove the node from the OPEN list it is set unactive
            OPEN[index_min_node,0] = 0
        else:
            #No path exists to the Target!!
            NoPath = 1 #Exits the loop!
        #End of index_min_node check
    #End of While Loop

    #Once algorithm has run The optimal path is generated by starting of at the
    #last node(if it is the target node) and then identifying its parent node
    #until it reaches the start node.This is the optimal path


    # Find the last node inserted in the closed list if everithing worked fine it should be the target node
    parent_x = CLOSED[-1,0]
    parent_y = CLOSED[-1,1]
    parent_z = CLOSED[-1,2]
    # Control if the last CLOSED node is the target one
    if ((parent_x == target[0] and parent_y == target[1] and parent_z == target[2])):
        #Traverse OPEN and determine the parent nodes
        #parent_x = int(OPEN[node_index(OPEN,xval,yval,zval),4])
        #parent_y = int(OPEN[node_index(OPEN,xval,yval,zval),5])
        #parent_z = int(OPEN[node_index(OPEN,xval,yval,zval),6])

        Optimal_path = [parent_x, parent_y, parent_z]

        while (parent_x != start[0] or parent_y != start[1] or parent_z != start[2]):
            Optimal_path = np.vstack([Optimal_path,[parent_x,parent_y,parent_z]])
            inode = node_index(OPEN, parent_x, parent_y, parent_z)
            parent_x = int(OPEN[inode,4])
            parent_y = int(OPEN[inode,5])
            parent_z = int(OPEN[inode,6])

        #end_time = tm.time() - start_time_all

        # Print a graphical output
        #visual_path(start,map,Optimal_path[1:np.shape(Optimal_path)[0]+1])

        Optimal_path = Optimal_path * dx
        for i in range(len(Optimal_path)):
            Optimal_path[i][2] = Optimal_path[i][2]/dx

        return Optimal_path

    else:
        print('\n Sorry, No path exists to the Target! \n')


#Robot Type
class RobotType(Enum):
    # in "Enum" you may access the instance by name or number
    circle = 0    # Round Robot
    rectangle = 1 # Rectuangular Robot - Skid Steering


""" Be AWARE!! SIMULATION INPUTS HERE!!!!!!! """
# Inputs Class--> Uses the robot type function
    # All the Robot inputs are stored here
class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 2.0  # [m/s]
        self.min_speed = 0.0  # [m/s]
        self.max_yawrate = 180.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 20  # [m/ss]
        self.max_dyawrate = 90.0 * math.pi / 180.0  # [rad/s^2]
        self.v_reso = 0.1  # [m/s]
        self.yawrate_reso = 5 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 1.0  # [s] How much into the future I am looking into
        # Bigger predict_time produces a longer trajectory at faster speed
        # Smaller predict_time produces a shorter trajectory at higher accelerations (but lower speeds)
        # The gains weights define which will be our objective
        self.to_goal_cost_gain = 0.1082 #each gain can vary from 0 to 1
        self.speed_cost_gain = 0.4771
        self.obstacle_cost_gain = 0.35
        self.path_cost_gain = 0.14
        self.dist_goal_cost_gain = 0
        self.robot_type = RobotType.circle
        # if goal_gain comparable to speed_gain the drone doesn't move

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 0.25  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check

    @property
    def robot_type(self):
        # Underscore indicates private variables in Python
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


def change_orientation(start,ptx,pty):

    global pose

    scarto = 0.1

    pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size = 10 )

    rate = rospy.Rate(10.0)

    target = np.array([ptx,pty])

    v = np.array([target[0]-start[0], target[1]-start[1]])

    degree = math.atan2(v[1],v[0])

    newdeg = start[2]

    msg_vel = Twist()

    turn_angle = degree-newdeg

    if(abs(turn_angle) < 0.1):
        return(start)
    
    if(turn_angle<-math.pi):
            turn_angle = 2*math.pi -(newdeg-degree)
    elif(turn_angle>math.pi):
            turn_angle = 2*math.pi - turn_angle

    time = abs(turn_angle)/0.35

    print(turn_angle, time)

    if turn_angle > 0:
        msg_vel.angular.z = 0.35
        publish_time(pub, msg_vel, time, rate)
        
    else:
        msg_vel.angular.z = -0.35
        publish_time(pub, msg_vel, time, rate)

    msg_vel.angular.z = 0

    publish_time(pub, msg_vel, 0.5, rate)

    rate.sleep()

    # print(newdeg)

    start[2] = degree*math.pi/180

    return(start)



def change_level(start, stop):

    global pose

    scarto = 0.025

    pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size = 10 )

    rate = rospy.Rate(10.0)

    msg_vel = Twist()

    if start > stop: 
        msg_vel.linear.z = -0.1
        while pose.position.z > stop + scarto:
            pub.publish(msg_vel)
            rate.sleep()
        msg_vel.linear.z = 0
        pub.publish(msg_vel)
    else:
        msg_vel.linear.z = 0.1
        while pose.position.z < stop - scarto:
            pub.publish(msg_vel)
            rate.sleep()
        msg_vel.linear.z = 0
        pub.publish(msg_vel)


def dwa_control(x, config, goal, ob, pt):
    """
    Dynamic Window Approach control:
        x = position vector (x,y,phi)
        config = Type of robot from Config() Class
        goal =
        ob =
    """

    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob, pt)

    return u, trajectory


def motion(x, u, dt):
    """
    motion model --> Kinematic Model
    We have two controls:
    u(0) = velocity
    u(1) = yaw_rate
    And we have 5 variables
    x(0:4) = x, y, yaw, ...?

    """

    x[2] += u[1] * dt    # Yaw angle [rad]
    x[0] += u[0] * math.cos(x[2]) * dt   # x [m]
    x[1] += u[0] * math.sin(x[2]) * dt   # y [m]
    x[3] = u[0] # V [m/s]
    x[4] = u[1] # omega [rad/s]

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x --> How much I am seeing in the future?
    x = initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yawrate, config.max_yawrate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_dyawrate * config.dt,
          x[4] + config.max_dyawrate * config.dt]

    #  [vmin, vmax, yaw_rate min, yaw_rate max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
         max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    traj = np.array(x)
    time = 0
    while time <= config.predict_time:

        x = motion(x, [v, y], config.dt)
        traj = np.vstack((traj, x))
        time += config.dt

    return traj


def calc_control_and_trajectory(x, dw, config, goal, ob, pt):
    """
    calculation final input with dynamic window
    x= initial state [x(m), y(m), yaw(rad), v(m/s), yaw_rate(rad/s)]
    dw=
    config=
    goal=
    on=
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    to_goal_cost = 0
    speed_cost = 0
    ob_cost = 0
    pt_cost = 0
    final_cost = 0

    # Apro il nuovo file per la scrittura
    txt = open("Log_cost", "a")

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_reso):
        for y in np.arange(dw[2], dw[3], config.yawrate_reso):

            trajectory = predict_trajectory(x_init, v, y, config)

            txt.write("\n---Velocita: " + str(v) + "\tVelocita angolare: " + str(y))

            # calc cost
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal, config)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)
            pt_cost = config.path_cost_gain * calc_path_cost(trajectory, pt)

            final_cost = to_goal_cost + speed_cost + ob_cost + pt_cost

            txt.write("\n    Goal: " + str(round(to_goal_cost, 4)) + "\tSpeed: " + str(round(speed_cost, 4)) + "\n    Obstacles: " + str(round(ob_cost, 4)) + "\tPath: " + str(round(pt_cost, 4)) + "\n    Total: " + str(round(final_cost, 4)))

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory

    txt.write("\n\nBest control:")
    txt.write("\n---Velocita: " + str(best_u[0]) + "\tVelocita angolare: " + str(best_u[1]) + "\n")
    txt.close()

    #print('to_goal, speed_cost, ob_cost, pt_cost')
    #print('{:.2f}, {:.2f}, {:.2f}, {:.2e} -> {:.4f}'.format(to_goal_cost,speed_cost,ob_cost,pt_cost,final_cost))

    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config):
    """
        calc obstacle cost inf: collision
    """
    ox = np.transpose(np.array(ob[0]))
    oy = np.transpose(np.array(ob[1]))
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        #local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if (r <= config.robot_radius).any():
            return float("inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK


def calc_to_goal_cost(trajectory, goal, config):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
    cost = (1 + abs(dx**2 + dy**2) / cost * config.dist_goal_cost_gain) * cost

    return cost


def calc_path_cost(trajectory, pt):
    """
        calc the cost outside the path
    """
    cost = 0
    for j in range(0,len(trajectory)):
        x = trajectory[j, 0]
        y = trajectory[j, 1]
        flag = 0
        for i in range(0,len(pt[0])):
            px = pt[0][i]
            py = pt[1][i]
            if (abs(x-px) < 0.15 * math.sqrt(2) and abs(y-py) < 0.15 * math.sqrt(2)):
                flag = 1
        if flag == 0:
            cost += 1

    return cost


#separates obstacles in levels (1m discr for z axis)
def separate_levels (ob, z_max, dx):
    levels = []
    for i in range(int(z_max * dx)+1):
        levels.append([[], []])

    for line in ob:
        z = int(line[2])
        if(not levels[z][0]):
            levels[z][0].insert(0,line[0])
            levels[z][1].insert(0,line[1])
        else:
            if(levels[z][0][-1] != line[0] or levels[z][1][-1] != line[1]):
                levels[z][0].append(line[0])
                levels[z][1].append(line[1])
    return levels


#returns list of waypoint sections with corresponding list of heights
def separate_wp (ptr):

    pt = ptr[::-1]

    wp_list = [[[],[]]]
    heights = []

    height = int(pt[0][2])
    heights.insert(0,height)

    wp_set = 0

    for wp in pt:

        if(height != int(wp[2])):
            wp_set += 1
            wp_list.append([[],[]])
            height = int(wp[2])
            heights.append(height)

        if(not wp_list[wp_set][0]):
            wp_list[wp_set][0].insert(0,wp[0])
            wp_list[wp_set][1].insert(0,wp[1])
        else:
            if(wp_list[wp_set][0][-1] != wp[0] or wp_list[wp_set][1][-1] != wp[1]):
                wp_list[wp_set][0].append(wp[0])
                wp_list[wp_set][1].append(wp[1])

    return(wp_list, heights)


def add_obstacle(ob, to_add):
    return np.transpose(np.unique(np.transpose(np.array([np.append(ob[0], to_add[0]), np.append(ob[1], to_add[1])])), axis = 0))


def dwa(start, target_dot, gx, gy, gz, ob, pt, z_max, size_step = 1, robot_type=RobotType.circle):
    global done
    global pose
    ob = np.transpose(np.unique(np.transpose(ob), axis=0))
    print(__file__ + " start!!")
    # Apro il file per il salvataggio degli stati
    file = open("Log_states.txt","w")
    file.write( "\t" + "x" + "\t\t" + "y" + "\t" + "theta" + "\t" + "V" + "\t\t" + "omega" )
    file.close()
    # Cancello il file precedente per il salvataggio dei costi
    txt = open("Log_cost", "w")
    txt.close()
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    #x = np.array([sx, sy, math.pi / 8.0, 0.0, 0.0])
    x = np.array([start[0], start[1], start[3], 0.0, 0.0])
    x[0] = pose.position.x
    x[1] = pose.position.y
    orientation_q = pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion (orientation_list)
    x[2] = yaw

    rospy.loginfo("acquire initial pose: %f %f %f" %(x[0], x[1], x[2]))
    #print("Posizione iniziale: " + str(x))

    # goal position [x(m), y(m)]
    """ FINAL OBJECTIVE""" # --> Maybe our final waypoint or the waypoints defined from A*
    goal = np.array([gx, gy])

    # print("Posizione target: " + str(pt[0][0])+ " " + str(pt[1][0]))
    x = change_orientation(x,pt[0][0],pt[1][0])
    # print("Posizione iniziale: " + str(x))

    """ OBSTACLES"""
    # obstacles [x(m) y(m), ....] --> Cells with Obstacles in x, y --> From Path Planning
    # We have to make sure we can feed a new matrix while computing (i.e. for new obstacles recognition)

    # input [forward speed, yaw_rate]

    config = Config() # Kinetic Definition Robot
    config.robot_type = robot_type
    trajectory = np.array(x)

    time_ellapsed = 0
    ttt = 0
    dist_to_goal = 0
    fermo = 0

    cmd = mavros_cmd()

    #Change the RF
    cmd.setMAVFrame()


    while not(done):

    # if np.linalg.norm(np.array([x[0],x[1],gz])-np.array([2.5, 5, 2])) <= 2:

    #        ob = add_obstacle(ob, [1.75, 4.25])
    #        ob = add_obstacle(ob, [2, 4.25])
    #        ob = add_obstacle(ob, [2.25, 4.25])
    #        ob = add_obstacle(ob, [2.5, 4.25])
    #        ob = add_obstacle(ob, [2.75, 4.25])

    #        ob = add_obstacle(ob, [1.75, 4.5])
    #        ob = add_obstacle(ob, [2, 4.5])
    #        ob = add_obstacle(ob, [2.25, 4.5])
    #        ob = add_obstacle(ob, [2.5, 4.5])
    #        ob = add_obstacle(ob, [2.75, 4.5])

    #        ob = add_obstacle(ob, [1.75, 4.75])
    #        ob = add_obstacle(ob, [2, 4.75])
    #        ob = add_obstacle(ob, [2.25, 4.75])
    #        ob = add_obstacle(ob, [2.5, 4.75])
    #        ob = add_obstacle(ob, [2.75, 4.75])

    #        ob = add_obstacle(ob, [1.75, 5])
    #        ob = add_obstacle(ob, [2, 5])
    #        ob = add_obstacle(ob, [2.25, 5])
    #        ob = add_obstacle(ob, [2.5, 5])
    #        ob = add_obstacle(ob, [2.75, 5])


        file = open("Log_states.txt","a")
        # u, predicted_trajectory is UNPACKING a vector returned by dwa_control
        # which is made by best_u (best control) and best trajectory found
        u, predicted_trajectory = dwa_control(x, config, goal, ob, pt) # --> How much in the future I am looking into?

        pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size = 10 )

        rate = rospy.Rate(10.0)

        # Messagge inizialization
        msg_vel = Twist()

        msg_vel.linear.x = u[0]/2
        msg_vel.angular.z = u[1]/2

        pub.publish(msg_vel)

        rate.sleep()

        x = motion(x, u, config.dt)  # simulate robot

        x[0] = pose.position.x
        x[1] = pose.position.y
        orientation_q = pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion (orientation_list)
        x[2] = yaw

        file.write("\n" + str(round(x[0], 4)) + "\t" + str(round(x[1], 4)) + "\t" + str(round(x[2], 4)) + "\t" + str(round(x[3], 4)) + "\t" + str(round(x[4], 4)) )
        file.close()
        # Possible control of position respert of the goal for A* recalculation
        if (x[3] > -0.001 and x[3] < 0.001) and (x[4] > -0.05 and x[4] < 0.05):
            fermo += 1
        else:
            fermo = 0
        trajectory = np.vstack((trajectory, x))  # store state history
        time_ellapsed += config.dt

        if (fermo * config.dt) > 1:
            fermo = 0
            #print(trajectory)
            print("Ricalcolo percorso")
            [dx, obst, x_max, y_max, z_max] = mapgenerator() #get map info (obstacles etc)
            obst = np.concatenate((obst,  [[int(o[0]/size_step), int(o[1]/size_step), int(gz/size_step)+1] for o in np.transpose(ob)]), axis=0)
            x = np.array([x[0]/dx, x[1]/dx, gz, x[2], 0, 0]) #x, y, z, angle, vx, vy --> set drone velocity to zero
            return True

        if show_animation:
            ttt +=config.dt
            plt.cla()
            plt.title("u(0)=V={:+3.2f} m/s, u(1)=dot(psi)={:+05.2f} rad/s, goal={:+05.2f} m".format(u[0],u[1],dist_to_goal))
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(pt[0], pt[1], "og")
            plt.plot(ob[0], ob[1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])

        '''
        if dist_to_goal <= 1 * config.robot_radius:
            print("Goal!!")
            file.close()
            break
        '''

        if dist_to_goal <= 1:
            #config.max_speed = 3
            #config.max_accel = 1
            config.dist_goal_cost_gain = 10
            #config.speed_cost_gain = 0.5
            if dist_to_goal <= 0.2:
                u[0] = 0
                u[1] = 0
                print("Goal!!")
                file.close()
                break
        else:
            #config.max_speed = 3
            #config.max_accel = 1
            #config.speed_cost_gain = 0.27
            config.dist_goal_cost_gain = 0

        # print("Dist to goal: " + str(dist_to_goal))
        # print("Dist_goal_cost_gain: " + str(config.dist_goal_cost_gain))

    if(np.linalg.norm(np.array([x[0],x[1],gz]) - np.array([target_dot[0]*size_step, target_dot[1]*size_step, target_dot[2]]))>0.2):

        x = np.array([x[0], x[1], 0, x[2], 0, 0])

        return False

    else:

        print("Done")

        return False


def inflate_obst(sizex, sizey, levels, dx):

    obstacles = []

    lev = 0

    for obst in levels:

        map = np.zeros([sizex, sizey])
        for i in range(len(obst[0])):
            map[int(obst[0][i]/dx)][int(obst[1][i]/dx)] = 1

        index = 0
        for o in map_to_obstacle_array(inflate_map(map)):
            obstacles.insert(index, [o[0], o[1], lev])
            index = index+1
        lev = lev + 1

    return np.array(obstacles)


def inflate_map(map):

    sizex= len(map)
    sizey= len(map[0])

    off1 = [0,0,1,1,1,-1,-1,-1]
    off2 = [1,-1,0,1,-1,0,1,-1]

    for x in range(sizex):
        for y in range(sizey):
            if(map[x][y]==1):
                for o in range(8):
                    if(x+off1[o]>=0 and x+off1[o]<sizex and y+off2[o]>=0 and y+off2[o]<sizey and map[x+off1[o]][y+off2[o]]==0):
                        map[x+off1[o]][y+off2[o]] = 8

    for x in range(sizex):
        for y in range(sizey):
            if(map[x][y]==8):
                map[x][y]=1

    return map


def map_to_obstacle_array(map):
    obst = [[],[]]
    for y in range(len(map)):
        for x in range(len(map[0])):
            if(map[y][x]==1):
                if(not(obst[0])):
                    obst[0].insert(0, y)
                    obst[1].insert(0, x)
                obst[0].append(y)
                obst[1].append(x)

    return np.transpose(np.array(obst))


"""if __name__ == '__main__':
    main(robot_type=RobotType.circle) """


def main(start_dot, target_dot, dx, obst, x_max, y_max, z_max):

    global flag_cl
    global flag_to
    global pose

    obst = np.unique(obst, axis = 0)

    levels = separate_levels(obst*dx, z_max, dx)

    pt = ast(start_dot, target_dot, dx, inflate_obst(x_max+2, y_max+2, list(levels), dx), x_max, y_max, z_max)

    [wp_list, heights] = separate_wp(pt)

    start = np.hstack((start_dot[0:3] * dx, start_dot[3:6]))

    for i in range(len(wp_list)):

        if(dwa(start, target_dot, wp_list[i][0][-1], wp_list[i][1][-1], heights[i], levels[heights[i]], wp_list[i], z_max, dx)):
            start[0] = int(pose.position.x / dx)
            start[1] = int(pose.position.y / dx)
            orientation_q = pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (roll, pitch, yaw) = euler_from_quaternion (orientation_list)
            start[3] = yaw
            return [True, start]
        else:
            start[0] = pose.position.x
            start[1] = pose.position.y
            orientation_q = pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (roll, pitch, yaw) = euler_from_quaternion (orientation_list)
            start[3] = yaw

            if i < len(wp_list)-1:
                print("livello attuale: " + str(heights[i]) + " livello successivo: " + str(heights[i+1]))

                change_level(pose.position.z, heights[i+1]+0.5)

    return [False, start]
                


if __name__ == '__main__':

    global flag_cl
    global flag_to
    global pose

    dx = 0.25
    x_max = 80
    y_max = 40
    z_max = 12
    obst = np.array([[0, 0, 0]])

    # Initiate node
    rospy.init_node('trajectory_node')

    pos = rospy.wait_for_message('/obst', PoseArray, timeout = 5)

    try:
        #rospy.loginfo(pos)
        
        for ob in pos.poses:
            obst = np.append(obst, [[ob.position.x, ob.position.y, ob.position.z]], axis = 0)

    except rospy.ROSException as e:
        rospy.loginfo('failed %s' %(e))

    # Flight mode object
    cmd = mavros_cmd()

    # Subscriber's inizialization to drone's local position
    sub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, pose_cb)

    # Publisher's inizialization to set the velocities
    pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size = 1 )

    # ROS loop rate 10 Hz
    rate = rospy.Rate(10.0)

    msg_vel = Twist()

    # Inizialization check
    print('Node initialized')

    # Set the GUIDED mode
    cmd.setGuidedMode()

    # Take off phase
    if flag_to != True:
        # Arm the motors
        cmd.setArm()

        # Execute takeoff
        cmd.setTakeoff()

        # Loop to consent the correct takeoff
        while flag_to is not True:
            pass

    # Confirmed Takeoff
    print('Takeoff phase termined')

    ast_flag = True

    start = np.array([10, 16, 0, 0, 0.0, 0.0])

    goals = [np.array([70, 32, 0]), np.array([16, 36, 0])]

    for target in goals:

        print(start, target)

        while(ast_flag == True):

            [ast_flag, start] = main(start_dot = start, target_dot = target, obst = obst, dx = dx, x_max = x_max+2, y_max = y_max+2, z_max = z_max)
        
        ast_flag = True

        start[0] = int(pose.position.x / dx)
        start[1] = int(pose.position.y / dx)
        orientation_q = pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion (orientation_list)
        start[3] = yaw

    
