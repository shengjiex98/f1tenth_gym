import time
import yaml
import gym
import numpy as np
from argparse import Namespace

from numba import njit

from pyglet.gl import GL_POINTS, GL_LINES

class PurePursuitPlanner:
    """
    Example Planner
    """
    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.max_reacquire = 20.

        self.path = None
        self.drawn_path = []

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        return
        #points = self.waypoints

        points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    def render_path(self, e):
        """
        Renders the path that the vehicle has followed
        """
        if self.path is None:
            return

        points = np.vstack((self.path[:, 0], self.path[:, 1])).T
        
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_path) < points.shape[0]:
                b = e.batch.add(1, GL_LINES, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_path.append(b)
            else:
                self.drawn_path[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        gives actuation given observation
        """
        if self.path is None:
            self.path = np.array([[pose_x, pose_y]])
        else:
            self.path = np.vstack((self.path, [pose_x, pose_y]))

        speed = 6.5
        tan_delta = ((2 * self.wheelbase / lookahead_distance ** 2) * 
                     (-np.sqrt(lookahead_distance ** 2 - pose_y ** 2) * np.sin(pose_theta) - pose_y * np.cos(pose_theta)))
        steering_angle = np.arctan(tan_delta)

        return speed, steering_angle

def main():
    work = {
        'mass': 3.463388126201571,
        'lf': 0.15597534362552312,
        'tlad': 1.5,
        'vgain': 0.90338203837889
    }
    
    with open('config.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    planner = PurePursuitPlanner(conf, 0.17145+0.15875)

    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        planner.render_path(env_renderer)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    env.add_render_callback(render_callback)
    
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    laptime = 0.0
    start = time.time()

    count = 0

    while not done:
        if not count:
            speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
        # count = (count + 1) % 6
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        env.render(mode='human')
        
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)

if __name__ == '__main__':
    main()
