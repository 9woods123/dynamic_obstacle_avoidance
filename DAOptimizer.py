# -*- coding: utf-8 -*-
import math
from wayPoint import wayPoint
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
from matplotlib.animation import FuncAnimation

###  dynamic obstacle model


class DynOb:
    
    def __init__(self,x,y,z=0,yaw=0,vel=0):
        self.x=x
        self.y=y
        self.z=z
        self.yaw=yaw
        self.vel=vel
    
    def updateOdom(self,x,y,z,yaw,vel):
        self.x=x
        self.y=y
        self.z=z
        self.yaw=yaw
        self.vel=vel


    def getPositionbyTime(self,d_t):
        ##linear model
        
        x=self.x + d_t*self.vel * math.cos(self.yaw)
        y=self.y + d_t* self.vel * math.sin(self.yaw)
        z=0
        # print("getPositionbyTime:", "x=",x, " y=",y, " d_t:",d_t)
        return np.array([x,y,z])
    



##Dynamic Avoidance Optimizer
class DAOptimizer:

    def __init__(self):

        self.__goal =None
        self.__start=None
        self.__init_path_points=None

        self.__obstacle=[]

        self.__smooth_cost_weight=1
        self.__collision_cost_weight=0.5
        self.__terminal_cost_weight=0.5

        self.__delta_t=0.25 ## delta_t between two near path points
        self.vel_=4    ## const motion

        self.safe_distance=3

        self.__static_obstacles=None


    def get_delta_time(self):
        return self.__delta_t
    
    def setGoal(self,way_point):
         if type(way_point) is not wayPoint:
             print("please setGoal with type wayPoint")
             return False
         self.__goal =way_point

    def setStart(self,way_point):
         if type(way_point) is not wayPoint:
             print("please setStart with type wayPoint")
             print("your type ",type(way_point))
             return False
         self.__start =way_point
    

    def addDyanmicObstacle(self,dyna_ob):
        if type(dyna_ob) is not DynOb:
             print("please addDyanmicObstacle with type DynOb")
             return False
        
        self.__obstacle.append(dyna_ob)



    def setObstacleMap(self,way_point_list):
        self.__static_obstacles = np.array([[point.x() ,point.y(),point.z() ]for point in way_point_list])
        
        

    def get_obstacle_position(self, time):
        if self.__obstacle is None:
            print("please setObstacle before using function get_obstacle_position()")
        self.__obstacle.get_obstacle_position_by_time(time)




    def init_path(self):
        if  self.__goal ==None or self.__start==None:
            print("please setGoal and setStart before init_path")
            return False
        ## 初始路径 匀速运动假设

        distance=wayPoint.getDistance(self.__start, self.__goal)
        num_steps = int(distance / self.vel_ / self.__delta_t)        # 计算时间步数,向下取整数。


        self.__init_path_points=[]                                                             # 生成路径点

        for curr_step in range(num_steps+1):
            curr_time=curr_step*self.__delta_t

            interpolated_point = self.__start + curr_time * self.vel_* ( self.__goal -  self.__start)/distance
            interpolated_point.set_time(curr_time)
            self.__init_path_points.append(interpolated_point)

        self.__init_path_points.append(self.__goal)
        self.__init_path_points[-1].set_time(distance / self.vel_)

        return self.__init_path_points  


    

    def smooth(self,init_path=None,obstacle_set=None):
        
        init_path=self.__init_path_points
        iterations = 0
        max_iterations =100
        path_length = len(init_path)

        if path_length < 5:
            return False

        # new_path = copy.deepcopy(init_path)
        # temp_path = copy.deepcopy(init_path)
        new_path = init_path.copy()
        # temp_path =  init_path.copy()
        alpha = 0.1  # You might need to adjust this based on your requirements
        
        total_cost=0

        while iterations < max_iterations:
            
            for i in range(2, path_length - 2):
                xim2 = np.array([new_path[i - 2].x(), new_path[i - 2].y(), new_path[i - 2].z()])
                xim1 = np.array([new_path[i - 1].x(), new_path[i - 1].y(), new_path[i - 1].z()])
                xi = np.array([new_path[i].x(), new_path[i].y(), new_path[i].z()])
                xip1 = np.array([new_path[i + 1].x(), new_path[i + 1].y(), new_path[i + 1].z()])
                xip2 = np.array([new_path[i + 2].x(), new_path[i + 2].y(), new_path[i + 2].z()])

                curr_time=new_path[i].get_time()
                # print("point time:",curr_time )

                smooth_grad,smooth_cost=self.smoothness_term(xim2, xim1, xi, xip1, xip2)
                collision_grad,collision_cost=self.obstacle_term(xi,curr_time)

                correction = self.__smooth_cost_weight*smooth_grad
                correction += self.__collision_cost_weight*collision_grad

                # total_cost= self.__smooth_cost_weight*smooth_cost+ \
                #                          self.__collision_cost_weight*collision_cost
                # print("total_cost=",total_cost)

                xi = xi - alpha * correction
                new_path[i]=wayPoint( xi[0], xi[1], xi[2],reach_time=curr_time)

                # temp_path[i]=wayPoint( xi[0], xi[1], xi[2])
            # print("collision_cost=",collision_cost)

            # new_path = temp_path
            ###  optimize two points at the end 
            xNim1 = np.array([new_path[path_length-1].x(), new_path[path_length-1].y(), new_path[path_length-1].z()])
            xNim2 = np.array([new_path[path_length-2].x(), new_path[path_length-2].y(), new_path[path_length-2].z()])
            collision_grad1,collision_cost1=self.obstacle_term(xNim1,new_path[path_length-1].get_time())
            collision_grad2,collision_cost2=self.obstacle_term(xNim2,new_path[path_length-2].get_time())

            terminal_grad,terminal_cost=self.terminal_term(xNim1)

            xNim1 = xNim1 - alpha *  self.__collision_cost_weight * collision_grad1 \
                            -alpha  * self.__terminal_cost_weight* terminal_grad
            xNim2 = xNim2 - alpha * collision_grad2

            new_path[path_length-1]=wayPoint( xNim1[0], xNim1[1], xNim1[2],reach_time=new_path[path_length-1].get_time())
            new_path[path_length-2]=wayPoint( xNim2[0], xNim2[1], xNim2[2],reach_time=new_path[path_length-2].get_time())


            iterations += 1
        
        return new_path



    def smoothness_term(self,xim2, xim1, xi, xip1, xip2):

        cost=np.linalg.norm((xip2-xip1) - (xip1 -xi))

        return xip2-4*xip1+6*xi-4*xim1+xim2, cost


    def terminal_term(self, endpoint):
        ## set the last point of  inil path  as  our expected terminal position
        terminal_point = np.array([ self.__init_path_points[-1].x(),  
                                            self.__init_path_points[-1].y(), 
                                            self.__init_path_points[-1].z()])
        
        cost=np.linalg.norm(terminal_point-endpoint)**2

        gradient=endpoint - terminal_point

        return gradient,cost



    def obstacle_term(self,xi,T):
        gradient = np.zeros(3)
        cost=0


        d_threshold=self.safe_distance

        if self.__obstacle is None:
            return gradient,cost


        for obstacle in self.__obstacle:

            obstacle_distance = np.linalg.norm(xi - obstacle.getPositionbyTime(T))

            if obstacle_distance > d_threshold:
                continue

            cost+=(obstacle_distance-d_threshold)**2

            obstacle_gradient = 2 * (obstacle_distance - d_threshold)*(xi -  obstacle.getPositionbyTime(T)) / (obstacle_distance)

            gradient += obstacle_gradient


        return gradient,cost



def plot_experiment(init_path,opt_path):

    init_path_coords_x = np.array([point.x() for point in init_path])
    init_path_coords_y = np.array([point.y() for point in init_path])

    opt_path_coords_x = np.array([point.x() for point in opt_path])
    opt_path_coords_y = np.array([point.y() for point in opt_path])


    # plot_circle((2, 2), 1)
    plt.figure(figsize=(12,12))
    plt.clf()
    plt.title("Planned Path",fontsize=25)
    plt.axis('equal')  # Set equal aspect ratio
    plt.plot(init_path_coords_x,init_path_coords_y, '--', linewidth=4)
    plt.plot(opt_path_coords_x,opt_path_coords_y, '.-', linewidth=3)


    # 调整标签的字体大小
    plt.xlabel('x (m)', fontsize=15)
    plt.ylabel('y (m)', fontsize=15)

    # 调整图例的字体大小
    plt.legend(['init_path', 'opt_path'], fontsize=20)


    plt.grid()
    # 调整刻度的字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.text(opt_path_coords_x[0], opt_path_coords_y[0], 'Start', fontsize=25, color='red', ha='left', va='bottom')
    plt.text(opt_path_coords_x[-1], opt_path_coords_y[-1]-0.5, 'Goal', fontsize=25, color='green', ha='right', va='bottom')
    plt.show()





def plot_gif(dyn_ob_list, opt_path):

    # 创建图像和轴
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 16)
    ax.set_ylim(-2, 16)

    # 初始化机器人的绘图对象
    robot, = ax.plot([], [], 'bo', markersize=10)  # 蓝色圆圈表示机器人

    # 初始化障碍物的绘图对象列表
    obstacles = []
    for _ in dyn_ob_list:
        obstacle, = ax.plot([], [], 'rs', markersize=10)  # 红色方块表示障碍物
        obstacles.append(obstacle)

    # 更新函数，用于生成动画的每一帧
    def update(frame):
        robot_position_x = opt_path[frame].x()
        robot_position_y = opt_path[frame].y()

        # 绘制机器人
        robot.set_data(robot_position_x, robot_position_y)

        # 更新每个障碍物的位置
        for i, obstacle in enumerate(obstacles):
            obstacle_x = dyn_ob_list[i].getPositionbyTime(opt_path[frame].get_time())[0]
            obstacle_y = dyn_ob_list[i].getPositionbyTime(opt_path[frame].get_time())[1]
            obstacle.set_data(obstacle_x, obstacle_y)

        return [robot] + obstacles

    # 创建动画
    ani = FuncAnimation(fig, update, frames=np.arange(0, len(opt_path), 1), interval=250)

    ani.save('gif_test.gif', writer='pillow')

    # 显示动画
    plt.show()




def main():

    algo_test=DAOptimizer()

    start_point = wayPoint(0, 0, 0)
    goal_point = wayPoint(14, 14, 0)

    ob_1 = DynOb(0,8,z=0,yaw=-math.pi/4, vel=4.5)
    # ob_2 = DynOb(14,12,z=0,yaw=1.2*math.pi, vel=1.5)
    ob_2 = DynOb(6,1,z=0,yaw=0.3*math.pi, vel=1.5)
    ob_3 = DynOb(20,7,z=0,yaw=0.9*math.pi, vel=3)

    algo_test.setStart(start_point)
    algo_test.setGoal(goal_point)
    algo_test.addDyanmicObstacle(ob_1)
    algo_test.addDyanmicObstacle(ob_2)
    algo_test.addDyanmicObstacle(ob_3)

    init_path_=algo_test.init_path()
    
    smooth_start_time = time.time()  # 
    opt_path_=algo_test.smooth()
    smooth_end_time = time.time()  # 
    print("waypoint size of path: ", len(opt_path_))
    print("time oost: ",smooth_end_time-smooth_start_time )
    plot_experiment(init_path_,opt_path_)



    plot_gif([ob_1,ob_2,ob_3],opt_path_)
    

if __name__ == "__main__":
    main()




# def plot_experiment(init_path,opt_path,static_obstacles):

# init_path_coords_x = np.array([point.x() for point in init_path])
# init_path_coords_y = np.array([point.y() for point in init_path])

# opt_path_coords_x = np.array([point.x() for point in opt_path])
# opt_path_coords_y = np.array([point.y() for point in opt_path])

# ob_coords_x = np.array([point.x() for point in static_obstacles])
# ob_coords_y = np.array([point.y() for point in static_obstacles])

# # plot_circle((2, 2), 1)
# plt.figure(figsize=(8,8))
# plt.clf()
# plt.title("Planned Path",fontsize=25)
# plt.axis('equal')  # Set equal aspect ratio
# plt.plot(init_path_coords_x,init_path_coords_y, '--', linewidth=4)

# plt.plot(opt_path_coords_x,opt_path_coords_y, '.-', linewidth=3)

#     # Plot circles for static obstacles
# for i in range(len(ob_coords_x)):
#     circle = Circle((ob_coords_x[i], ob_coords_y[i]), radius=OB_R, fc='red', ec='black')
#     plt.gca().add_patch(circle)


# # 调整标签的字体大小
# plt.xlabel('x (m)', fontsize=25)
# plt.ylabel('y (m)', fontsize=25)

# # 调整图例的字体大小
# plt.legend(['init_path', 'opt_path'], fontsize=20)


# plt.grid()
# # 调整刻度的字体大小
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.text(opt_path_coords_x[0], opt_path_coords_y[0], 'Start', fontsize=25, color='red', ha='left', va='bottom')
# plt.text(opt_path_coords_x[-1], opt_path_coords_y[-1]-0.5, 'Goal', fontsize=25, color='green', ha='right', va='bottom')
# plt.savefig('data_file/init_path.pdf',tight_layout='tight', bbox_inches='tight')
# plt.show()