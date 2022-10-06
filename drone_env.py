import airsim
import gym
import numpy as np
import random
import time


class AirSimDroneEnv(gym.Env):
    def __init__(self, ip_address, image_shape):
        self.image_shape = image_shape
        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        #self.action_space = gym.spaces.Box(np.array([-5,-5,-5]).astype(np.float32), np.array([+5,+5,+5]).astype(np.float32))
        self.action_space = gym.spaces.Discrete(9)
        self.info = {"collision": False}
        
        self.win = 0
        self.collision_time = 0
        self.random_start = True
        self.setup_flight()

    def step(self, action):
        self.do_action(action)
        obs, info = self.get_obs()
        reward, done = self.compute_reward()
        return obs, reward, done, info

    def reset(self):
        self.setup_flight()
        obs, _ = self.get_obs()
        return obs

    def render(self):
        return self.get_obs()

    def setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        self.drone.moveToZAsync(-1, 1)
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp
        
        # ---------------------------------------------------------------------------
        
        self.target_pos = [80,random.randint(-50,50),5]
        
        pose = airsim.Pose(airsim.Vector3r(0, 0, 0))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        #self.a=1
        self.target_dist_prev = abs(np.linalg.norm(np.array([0,0,0]) - self.target_pos))
        self.startpose = self.target_dist_prev
        # ---------------------------------------------------------------------------
        
    def do_action(self, select_action):
        
        #if select_action is not None:
        #    vx = select_action[0]
        #    vy = select_action[1]
        #    vz = select_action[2]
        
        speed = 1.0
        if select_action == 0:
        	vy, vz = (-speed, -speed)
        elif select_action == 1:
        	vy, vz = (0, -speed)
        elif select_action == 2:
        	vy, vz = (speed, -speed)
        elif select_action == 3:
        	vy, vz = (-speed, 0)
        elif select_action == 4:
        	vy, vz = (0, 0)
        elif select_action == 5:
        	vy, vz = (speed, 0)
        elif select_action == 6:
        	vy, vz = (-speed, speed)
        elif select_action == 7:
        	vy, vz = (0, speed)
        else:
        	vy, vz = (speed, speed)
        
        #self.drone.moveByVelocityBodyFrameAsync(float(vx), float(vy), float(vz), duration=1).join()
        self.drone.moveByVelocityBodyFrameAsync(speed, float(vy), float(vz), duration=1).join()
        self.drone.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=0.2)
        
        x,y,z = self.drone.simGetVehiclePose().position
        if x>=75:
        	self.drone.moveToPositionAsync(self.target_pos[0],self.target_pos[1],self.target_pos[2], 1.0).join()
        	#self.a=0
        
    def get_obs(self):
        self.info["collision"] = self.is_collision()
        obs = self.get_rgb_image()
        return obs, self.info

    def compute_reward(self):
        reward = 0
        done = 0
        x,y,z = self.drone.simGetVehiclePose().position
        target_dist_curr = np.linalg.norm(np.array([x,y,z]) - self.target_pos)
	
        if self.is_collision() :
            reward =-100
            done = 1
            
        elif x>=30.0 or y < -10.0 or y > 40.0 or z < -30 or z>-15:
            reward = -100
            done = 1
        
        else:
        	reward +=(self.target_dist_prev - target_dist_curr)*5
        	self.target_dist_prev = target_dist_curr
        	if y>40.0:
        		reward -= (y-40.0)*2
        	if z<-30.5:
        		reward += (z+30.0)*2
        		
        	if abs(target_dist_curr) <= 5.0:
        		reward +=200
        		done = 1
        	elif abs(target_dist_curr) <= 15.0:
        		reward +=(100 - abs(target_dist_curr))
       
        return reward, done


    def is_collision(self):
        current_collision_time = self.drone.simGetCollisionInfo().time_stamp
        return True if current_collision_time != self.collision_time else False
    
    def get_rgb_image(self):
        rgb_image_request = airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
        responses = self.drone.simGetImages([rgb_image_request])
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3)) 

        try:
            return img2d.reshape(self.image_shape)
        except:
            return np.zeros((self.image_shape))
        
