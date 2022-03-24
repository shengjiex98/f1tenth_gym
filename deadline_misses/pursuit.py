import collections
import time
import yaml
import gym
import numpy as np
from argparse import Namespace

from numba import njit

from pyglet.gl import GL_POINTS, GL_LINES, GL_LINE_STRIP

class PurePursuitPlanner:
    """
    Example Planner
    """
    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.max_reacquire = 20

        # Used for deadline misses handling
        self.last = 0
        self.miss_last = False
        self.saved_state = None

        self.path = collections.deque(maxlen=500)
        self.drawn_path = []

    def render_path(self, e):
        """
        Renders the path that the vehicle has followed
        """
        if len(self.path) == 0:
            return

        # points = [self.path[x] for x in range(0, len(self.path), 5)]
        points = self.path
        points = np.array(points)
        
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_path) < points.shape[0]:
                b = e.batch.add(1, GL_LINE_STRIP, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_path.append(b)
            else:
                self.drawn_path[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, miss=False, hold=True, kill=True):
        """
        gives actuation given observation
        """
        # Save path for rendering
        self.path.append([pose_x, pose_y])

        # Calculates the steering angle
        tan_delta = ((2 * self.wheelbase / lookahead_distance**2) * 
                     (-np.sqrt(lookahead_distance**2 - pose_y**2) * np.sin(pose_theta) - pose_y * np.cos(pose_theta)))
        steering_angle = np.arctan(tan_delta)

        # Handle misses
        if kill:
            if miss:
                result = self.last if hold else 0
            else:
                result = steering_angle
        else:
            # Skip-next
            if miss:
                result = self.last if hold else 0
                if not self.miss_last:
                    # HM
                    self.saved_state = steering_angle
            else:
                if not self.miss_last:
                    # HH
                    result = steering_angle
                else:
                    # MH
                    result = self.saved_state
        self.last = result
        self.miss_last = miss

        return 6.5, result

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
            speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], 
                                        work['tlad'], miss=False, hold=True, kill=True)
        count = (count + 1) % 17
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        env.render(mode='human')
        
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)

if __name__ == '__main__':
    main()
