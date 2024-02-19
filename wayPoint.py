# -*- coding: utf-8 -*-

import math
import numpy as np

class wayPoint(object):
    def __init__(self, x, y, z=0.0, vel_x=0.0, vel_y=0.0, vel_z=0.0, roll=0.0, pitch=0.0, yaw=0.0,reach_time=0):
        self.__x = x
        self.__y = y
        self.__z = z
        self.__vel_x = vel_x
        self.__vel_y = vel_y
        self.__vel_z = vel_z
        self.__roll = roll  ## rotate along the x-axis
        self.__pitch = pitch  ## rotate along the y-axis
        self.__yaw = yaw  ## rotate along the z-axis
        self.__t=reach_time

    def x(self):
        return self.__x

    def y(self):
        return self.__y

    def z(self):
        return self.__z

    def vel_x(self):
        return self.__vel_x

    def vel_y(self):
        return self.__vel_y

    def vel_z(self):
        return self.__vel_z

    def roll(self):
        return self.__roll

    def pitch(self):
        return self.__pitch

    def yaw(self):
        return self.__yaw


    def set_time(self, t_):
        self.__t=t_

    def get_time(self):
        return  self.__t

    def get_reach_time(self):
        return self.__t
    

    def unit_vector(self, other_point):
        """
        计算当前点到另一点的单位向量
        """
        vector = np.array([other_point.x() - self.__x, other_point.y() - self.__y, other_point.z() - self.__z])
        length = np.linalg.norm(vector)
        # 避免除以零
        if length == 0:
            return np.array([0.0, 0.0, 0.0])
            
        return vector / length


    @staticmethod
    def getDistance(from_point, to_point):
        return math.sqrt(
            (from_point.x() - to_point.x())**2 +
            (from_point.y() - to_point.y())**2 +
            (from_point.z() - to_point.z())**2
        )

    # 定义加法运算符
    def __add__(self, other):
        if isinstance(other, wayPoint):
            return wayPoint(self.x() + other.x(), self.y() + other.y(), self.z() + other.z())
        else:
            raise TypeError("Unsupported operand type")

    # 定义减法运算符
    def __sub__(self, other):
        if isinstance(other, wayPoint):
            return wayPoint(self.x() - other.x(), self.y() - other.y(), self.z() - other.z())
        else:
            raise TypeError("Unsupported operand type")

    # 定义乘法运算符
    def __mul__(self, scalar):
        return wayPoint(self.x() * scalar, self.y() * scalar, self.z() * scalar)
    
    # 定义右乘法运算符
    def __rmul__(self, scalar):
        return self.__mul__(scalar)
    
    # 定义除法运算符
    def __div__(self, scalar):
        if scalar != 0:
            return wayPoint(self.x() / scalar, self.y() / scalar, self.z() / scalar)
        else:
            raise ValueError("Division by zero")
        
    # 定义除法运算符
    def __truediv__(self, scalar):
        if scalar != 0:
            return wayPoint(self.x() / scalar, self.y() / scalar, self.z() / scalar)
        else:
            raise ValueError("Division by zero")

if __name__ == "__main__":

    # 示例使用
    point1 = wayPoint(1, 2, 3)
    point2 = wayPoint(4, 5, 6)

    result_add = point1 + point2
    result_sub = point1 - point2
    result_mul = point1 * 2
    result_mul = 2*point1 

    result_div = point1 / 2.0

    print(result_add.x(), result_add.y(), result_add.z())  # 输出: 5 7 9
    print(result_sub.x(), result_sub.y(), result_sub.z())  # 输出: -3 -3 -3
    print(result_mul.x(), result_mul.y(), result_mul.z())  # 输出: 2 4 6
    print(result_div.x(), result_div.y(), result_div.z())  # 输出: 0.5 1.0 1.5
