import collections
import itertools
import time
import yaml
import gym
import numpy as np
from argparse import Namespace

from pyglet.gl import GL_POINTS, GL_LINES, GL_LINE_STRIP

class PurePursuitPlanner:
    """
    Example Planner
    """
    def __init__(self, conf, wb, hold=True, kill=True):
        self.wheelbase = wb
        self.conf = conf
        self.hold = hold
        self.kill = kill

        # Used for deadline misses handling
        self.last = 0
        self.hit_last = True
        self.saved_state = None

    def render_path(self, e):
        """
        Renders the path that the vehicle has followed
        """

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, hit=True):
        """
        gives actuation given observation
        """
        # Calculates the steering angle
        # Nonlinear controller
        # tan_delta = ((2 * self.wheelbase / lookahead_distance**2) * 
        #              (-np.sqrt(lookahead_distance**2 - pose_y**2) * np.sin(pose_theta) - pose_y * np.cos(pose_theta)))
        # steering_angle = np.arctan(tan_delta)
        #
        # Linearized controller
        if (pose_theta > np.pi):
            pose_theta -= 2 * np.pi
        steering_angle = -(2 * self.wheelbase / lookahead_distance) * (pose_y / lookahead_distance + pose_theta)

        # Handle misses
        if self.kill:
            if not hit:
                result = self.last if self.hold else 0
            else:
                result = steering_angle
        else:
            # Skip-next
            if not hit:
                result = self.last if self.hold else 0
                if self.hit_last:
                    # HM
                    self.saved_state = steering_angle
            else:
                if self.hit_last:
                    # HH
                    result = steering_angle
                else:
                    # MH
                    result = self.saved_state
        self.last = result
        self.hit_last = hit

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

    planner = PurePursuitPlanner(conf, 0.17145+0.15875, hold=True, kill=True)

    # path = collections.deque(maxlen=500)
    path = []
    drawn_path = []

    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer
        render_size = 1000

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - render_size
        e.right = right + render_size
        e.top = top + render_size
        e.bottom = bottom - render_size

        if len(path) == 0:
            return

        # points = [self.path[x] for x in range(0, len(self.path), 5)]
        points = path
        points = np.array(points)
        points = points[:, 0:2]
        
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(drawn_path) < points.shape[0]:
                b = e.batch.add(1, GL_LINE_STRIP, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                drawn_path.append(b)
            else:
                drawn_path[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    env.add_render_callback(render_callback)
    
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta, conf.sv]]))
    env.render()

    laptime = 0.0
    start = time.time()

    period = 2     # in centiseconds
    num_hit = 1
    num_miss = 0

    # hit_pattern = itertools.cycle([True] * num_hit + [False] * num_miss)
    hit_pattern = itertools.cycle([False] * num_miss + [True] * num_hit)

    delayed = 6.5, 0

    time_span = 5
    count = 0
    for i in range(0, time_span * 100):

        if not count:
            # speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], 
            #                             work['tlad'], hit=next(hit_pattern))
            speed, steer = delayed
            delayed = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], 
                                   work['tlad'], hit=next(hit_pattern))
        
        # Save path for rendering
        path.append([obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], steer])

        count = (count + 1) % period
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        env.render(mode='human')

        # time.sleep(.02)
    
    with open(f"p{period}_{num_hit}hit{num_miss}miss.tsv", "w") as file:
        file.write("x\ty\ttheta\tsteer\n")
        file.writelines(map(lambda ls : f"{ls[0]}\t{ls[1]}\t{ls[2]}\t{ls[3]}\n", list(path)[::period]))
        
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)

if __name__ == '__main__':
    main()
