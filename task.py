'''
code provided in part by Udacity among the course materials of the Nanodegree Machine Learning Engineer

'''

import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 1

        self.state_size = self.action_repeat * 3
        self.action_low = 0
        self.action_high = 900
        #self.action_high = 450
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #1 reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #2 reward = -(np.log((abs(self.sim.pose[:3] - self.target_pos)).sum()))
        #3 reward = -(((abs(self.sim.pose[:3] - self.target_pos)).sum())**2)
        #sig2 = 2.5
        #sig2 = .5
        #sig2 = 10.
        #sig2 = .1
        #4 reward = 10000*(np.exp(-(((abs(self.sim.pose[:3] - self.target_pos)).sum())**2)/(2*(sig2))))
        #5 reward = -(np.linalg.norm(self.sim.pose[:3] - self.target_pos))
        #reward = -np.log(0.001*np.linalg.norm(self.sim.pose[:3] - self.target_pos))
        #reward = np.tanh(1 - 0.03*np.linalg.norm(self.sim.pose[:3] - self.target_pos))
        #reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        #reward = -.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #reward -= 0.1*(abs(self.sim.v)).sum() 
        #reward -= 1.1*(abs(self.sim.angular_v)).sum() 
        #reward = np.tanh((1 - 0.005*(abs(self.sim.pose[:3] - self.target_pos))-0.01*(abs(self.sim.v))-0.005*(abs(self.sim.angular_v))).sum())
        #reward = -np.log((1.+0.005*(abs(self.sim.pose[:3] - self.target_pos))+0.003*(abs(self.sim.v))+0.003*(abs(self.sim.angular_v))).sum())
        #reward = +.3*((self.sim.pose[2] - self.target_pos[2]))
        #reward = np.tanh(+.003*(self.sim.pose[2] - self.target_pos[2])-.0015*(abs(self.sim.pose[:2] - self.target_pos[:2])).sum())
        
        #sig2 = 2.
        #reward = np.exp(  -(  (  np.linalg.norm(self.sim.pose[:3] - self.target_pos)  )**1  )  /  (1.*(sig2))   )
        
        #reward = .3*(self.sim.pose[2] - self.target_pos[2])
        #reward -= .15*(abs(self.sim.pose[:2] - self.target_pos[:2])).sum()
        
        reward = -np.log(1.-.03*(self.sim.pose[2] - self.target_pos[2])+.015*(abs(self.sim.pose[:2] - self.target_pos[:2])).sum())
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            #pose_all.append(self.sim.pose)
            pose_all.append(self.sim.pose[:3])
        next_state = np.concatenate(pose_all)
        
        #if np.linalg.norm(self.sim.pose[:3] - self.target_pos) < 0.1:
            #reward += 10
            #done = True
            #if np.linalg.norm(self.sim.pose[:3] - self.target_pos) < 0.05:
                #reward += 100
                #done = True
        #if done and self.sim.pose[2] < self.target_pos[2]:
            #reward -= 0.
        if self.sim.pose[2] >= self.target_pos[2]:
            reward += 0.
            #reward += 10.
            done = True
        
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #state = np.concatenate([self.sim.pose] * self.action_repeat) 
        state = np.concatenate([self.sim.pose[:3]] * self.action_repeat)
        return state


class TaskLanding():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=3., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) for the landing
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 1

        self.state_size = self.action_repeat * 4
        self.action_low = 0
        self.action_high = 400
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #1 reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()-.1*(abs(self.sim.pose[4]))-.3*(abs(self.sim.v)).sum()-.3*(abs(self.sim.angular_v)).sum()
        #2 reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()-0.*(abs(self.sim.pose[4]))-.0*(abs(self.sim.v)).sum()-.0*(abs(self.sim.angular_v)).sum()
        #3 reward = 1.-.3*(((abs(self.sim.pose[:3] - self.target_pos)).sum())**2) -0.*(abs(self.sim.pose[4]))-.0*(abs(self.sim.v)).sum()-.0*(abs(self.sim.angular_v)).sum()
        #4 reward = 1.-.3*(np.log((abs(self.sim.pose[:3] - self.target_pos)).sum())) -0.*(abs(self.sim.pose[4]))-.0*(abs(self.sim.v)).sum()-.0*(abs(self.sim.angular_v)).sum()
        #sig2 = 1.
        #sig2 = 5.
        #sig2 = 2.5
        #5 reward = (np.exp(-(((abs(self.sim.pose[:3] - self.target_pos)).sum())**2)/(2*(sig2)))) -0.*(abs(self.sim.pose[4]))-.0*(abs(self.sim.v)).sum()-.0*(abs(self.sim.angular_v)).sum()
        #6 reward = -(np.log((abs(self.sim.pose[:3] - self.target_pos)).sum())) -0.*(abs(self.sim.pose[4]))-.0*(abs(self.sim.v)).sum()-.0*(abs(self.sim.angular_v)).sum()
        #7 reward = 100*(np.exp(-(((abs(self.sim.pose[:3] - self.target_pos)).sum())**2)/(2*(sig2)))) -0.*(abs(self.sim.pose[4]))-.0*(abs(self.sim.v)).sum()-.0*(abs(self.sim.angular_v)).sum()
        #8 reward = -(((abs(self.sim.pose[:3] - self.target_pos)).sum())**2) -0.1*(abs(self.sim.pose[4]))-0.1*(abs(self.sim.v)).sum()-0.1*(abs(self.sim.angular_v)).sum()
        #reward = -np.log(1.+0.001*(abs(self.sim.pose[2] - self.target_pos[2])) +0.001*(np.linalg.norm(self.sim.v)) ) 
        #reward -= np.log(1.+0.0015*(abs(self.sim.pose[4])) ) 
        #reward -= np.log(1.+0.03*(abs(self.sim.v)).sum() ) 
        #reward -= np.log(1.+.0015*(abs(self.sim.angular_v)).sum() ) 
        #reward = 0.5/((self.sim.pose[2] - self.target_pos[2]))
        #reward += np.log(1.-0.005*abs(self.sim.v[2]))
        #reward = -np.log(1.+0.003*(abs(self.sim.pose[:3] - self.target_pos)).sum() +0.003*(abs(self.sim.v)).sum() )
        #reward = 1./abs(self.sim.pose[2] - self.target_pos[2]) if abs(self.sim.pose[2] - self.target_pos[2]) else 10.
        #reward -= 0.1*np.linalg.norm(self.sim.v)
        #reward = -np.log(1. +0.03*(self.sim.pose[2] - self.target_pos[2]) )
        #reward -= np.log(1. +0.03*abs(self.sim.v[2]) )
        
        #sig2 = 100.
        #reward = np.exp(  -(  (  abs(self.sim.pose[2] - self.target_pos[2])   )**1  )  /  (1.*(sig2))   )
        #reward += np.exp(  -(  (  abs(self.sim.v[2])  )**1  )  /  (1.*(sig2))   )
        
        #reward = np.tanh(-0.003*(self.sim.pose[2] - self.target_pos[2]) )
        #reward += np.tanh(-0.003*abs(self.sim.v[2]) )
        #reward = -3.*(self.sim.pose[2] - self.target_pos[2]) -1.*abs(self.sim.v[2])
        
        reward = -5.*(abs(self.sim.pose[:3] - self.target_pos[:3]).sum())
        reward -= 5./(self.sim.v[2])
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(list(self.sim.pose[:3])+list([self.sim.v[2]]))
        next_state = np.concatenate(pose_all)
        
        #if (self.sim.pose[2] - self.target_pos[2]) < 0.:
            #reward -= 1.
        #if abs(self.sim.pose[2] - self.target_pos[2]) <= 0.01 and (abs(self.sim.v)).sum() < 0.1:
        #if abs(self.sim.pose[2] - self.target_pos[2]) <= 0.1 and (np.linalg.norm(self.sim.v)) <= 0.1:
        if (self.sim.pose[2] - self.target_pos[2]) <= 0.:
            reward += 1.
            done = True
            
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([np.concatenate((self.sim.pose[:3],[self.sim.v[2]]))] * self.action_repeat) 
        return state


