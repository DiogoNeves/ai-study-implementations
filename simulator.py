# !/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from itertools import izip_longest


"""
This was my implementation for the Artificial Intelligence: A Modern Approach exercise.
Chapter 2 exercise 8 and 9.

This file contains code to create, setup and run a simulation environment of a simple vacuum cleaner problem.
Modular so we can experiment different agents, performance functions, maps etc...

What I'd do better...
I'd manage the actions in the Agent itself and create try_action method in the environment, called by an actuator
instead of an Agent directly.

This would represent better, the structure of the reality we are analysing in the exercise.

I'm also not sure about how I implemented the sensors, they receive too much of the world, I'd replace this into
queries.

I'm happy I didn't spend long on it... lets move on :)
"""


Position = namedtuple('Position', ['x', 'y'])


def dust_sensor(world_map, position):
    return world_map.has_flag(Environment.STATES['DIRT'], position)


def simple_left_right_location_sensor(world_map, position):
    return 'left' if ((position.x + 1) / world_map.width) < 1 else 'right'


class Agent(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        """Initialise a new Agent."""
        pass

    @abstractmethod
    def step(self, time_step, state, position):
        pass


class SimpleReflexAgent(Agent):
    def step(self, time_step, world_map, position):
        if dust_sensor(world_map, position):
            return Environment.ACTIONS['SUCK']
        location = simple_left_right_location_sensor(world_map, position)
        if location == 'left':
            return Environment.ACTIONS['MOVE_RIGHT']
        else:
            return Environment.ACTIONS['MOVE_LEFT']


class Map:
    __metaclass__ = ABCMeta

    def __init__(self):
        """Initialise a new map."""
        self._width = 0
        self._height = 0
        self._state = self.get_initial_state()

    @abstractmethod
    def get_initial_state(self):
        """Return a list with the initial state at each location.

        States are bitmaps of all possible states:
            0: Nothing there
            1: Wall
            2: Dirt
        """
        pass

    def state(self, position):
        return self._state[self._get_index_from(position)]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def size(self):
        return self.width * self.height

    @staticmethod
    def is_flag_in_state(state, flag):
        return (flag & state) == flag

    def has_flag(self, flag, position):
        """Check if the flag is already set at the position."""
        return Map.is_flag_in_state(self.state(position), flag)

    def move_flag(self, flag, start, destination):
        """Remove the flag from start and add to destination.

        Args:
            flag (int) - Flag to add.
            start (Position) - Position to remove from.
            destination (Position) - Position to add to.
        """
        self.remove_flag(flag, start) \
            .add_flag(flag, destination)
        return self

    def add_flag(self, flag, position):
        """Add flag to the state at position."""
        self._state[self._get_index_from(position)] |= flag
        return self

    def remove_flag(self, flag, position):
        """Remove flag from the state at position."""
        self._state[self._get_index_from(position)] &= ~flag
        return self

    def valid_position(self, position):
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def count_flag(self, flag):
        return sum([1 for s in self._state if Map.is_flag_in_state(s, flag)])

    def _get_index_from(self, position):
        """Convert the position (x, y) to the state list index."""
        return position.x + position.y * self.width

    def __repr__(self):
        def grouper(iterable, n):
            """Collect data into fixed-length chunks or blocks."""
            args = [iter(iterable)] * n
            return izip_longest(*args)

        separator = '-%s-\n' % ('-' * (self.width * 2 - 1))
        result = separator
        for state_row in grouper(self._state, self.width):
            result += '|%s|\n' % ' '.join([str(s) for s in state_row])
        result += separator
        return result


class TwoLocationMap(Map):
    def __init__(self):
        super(TwoLocationMap, self).__init__()
        self._width = 2
        self._height = 1

    def get_initial_state(self):
        return [0, 2]


class Environment(object):
    """Class responsible for managing environments. Assumes all input is valid.

    Positions are base 0.
    States are bitmaps of all states at the location.
    Only 1 agent can be at a location at any point in time.

    Agents are updated in the order they were added.
    """

    STATES = {
        'NOTHING': 0,
        'WALL': 1,
        'DIRT': 2,
        'AGENT': 4
    }

    ACTIONS = {
        'NOTHING': 0,
        'MOVE_RIGHT': 1,
        'MOVE_LEFT': 2,
        'MOVE_UP': 3,
        'MOVE_DOWN': 4,
        'SUCK': 5
    }

    _AgentInfo = namedtuple('AgentInfo', ['agent', 'position', 'score', 'last_action'])

    def __init__(self, world_map, dirt_distribution_func, performance_measure_func):
        """Initialise a new Environment with the given map.

        Args:
            map (Map) - Map of the environment that will define the initial positions of different objects.
            dirt_distribution_func (func(state)) - Function that distributes dirt.
            performance_measure_func (func(state, agent, score)) - Measures performance of the agents.
        """
        self._agents = []
        self._map = world_map
        self._dirt_distribution = dirt_distribution_func
        self._performance_measure = performance_measure_func
        self._time_step = -1

    def add_agent(self, agent, position):
        """Add an Agent to the given position.

        Args:
            agent (Agent) - Agent to add.
            position (Position) - Position where to add the agent.

        Return:
            True if added successfully, False otherwise.
        """
        agent_info = Environment._AgentInfo(agent, position, 0.0, None)
        if self._can_add_flag(Environment.STATES['AGENT'], position):
            self._agents.append(agent_info)
            self._map.add_flag(Environment.STATES['AGENT'], position)
            return True
        return False

    def run(self, steps, verbose=False):
        for s in range(steps):
            self.step()
            if verbose:
                print self

        print 'Finished in %d time steps!' % (self._time_step + 1)
        print 'Agents:\n'
        for agent_info in self._agents:
            print '%s -> %f\n' % (type(agent_info.agent).__name__, agent_info.score)

    def step(self):
        """Run another time step."""
        self._time_step += 1
        new_agents = []
        for agent_info in self._agents:
            action = agent_info.agent.step(self._time_step, self._map, agent_info.position)
            new_agent_info = self._update_state(agent_info, action)
            score = self._performance_measure(self._map, agent_info)
            new_agents.append(Environment._AgentInfo(new_agent_info.agent, new_agent_info.position, score, action))
        self._agents = new_agents
        self._dirt_distribution(self._map)

    def _update_state(self, agent_info, action):
        """Update the state based on an agent's action."""

        def safe_move(delta_x, delta_y):
            end = Position(agent_info.position.x + delta_x, agent_info.position.y + delta_y)
            if self._can_add_flag(Environment.STATES['AGENT'], end):
                self._map.move_flag(Environment.STATES['AGENT'], agent_info.position, end)
                return Environment._AgentInfo(agent_info.agent, end, agent_info.score, agent_info.last_action)
            return agent_info

        if action == Environment.ACTIONS['NOTHING']:
            return agent_info

        if action == Environment.ACTIONS['SUCK']:
            self._map.remove_flag(Environment.STATES['DIRT'], agent_info.position)
            return agent_info
        elif action == Environment.ACTIONS['MOVE_DOWN']:
            return safe_move(0, 1)
        elif action == Environment.ACTIONS['MOVE_UP']:
            return safe_move(0, -1)
        elif action == Environment.ACTIONS['MOVE_LEFT']:
            return safe_move(-1, 0)
        elif action == Environment.ACTIONS['MOVE_RIGHT']:
            return safe_move(1, 0)
        return agent_info

    def _can_add_flag(self, flag, position):
        """Test if the flag can be added to the state at position.

        * Anything can be added to an empty location.
        * Walls can only be added to empty locations.
        * Agents can't be added to the same location as another agent.
        * Dirt can't be added to a location already dirty.
        """
        if not self._map.valid_position(position):
            return False
        if self._map.state(position) == Environment.STATES['NOTHING']:
            return True
        elif self._map.state(position) == Environment.STATES['WALL']:
            return False
        elif flag == Environment.STATES['AGENT'] and not self._map.has_flag(flag, position):
            return True
        elif flag == Environment.STATES['DIRT'] and not self._map.has_flag(flag, position):
            return True
        return False

    def __repr__(self):
        def action_name(action):
            if action is None:
                return 'None'
            return [k for k, v in Environment.ACTIONS.iteritems() if v == action][0]

        result = '=Environment=\n'
        result += 'Time Step: %d\n' % self._time_step
        result += 'Agents:\n'
        for agent_info in self._agents:
            result += '%s -> %f (Last Action: %s)\n' % (
                type(agent_info.agent).__name__, agent_info.score, action_name(agent_info.last_action))
        result += repr(self._map)
        return result


def perf_count_clean_spots(world_map, agent_info):
    return agent_info.score + (world_map.size - world_map.count_flag(Environment.STATES['DIRT']))


def perf_count_clean_with_penalty(world_map, agent_info):
    _PENALTY = 1
    return perf_count_clean_spots(world_map, agent_info) - (_PENALTY if 1 <= agent_info.last_action <= 4 else 0)


def add_dirt_to_same_position(world_map):
    return world_map.add_flag(Environment.STATES['DIRT'], Position(0, 0))


if __name__ == '__main__':
    env = Environment(TwoLocationMap(), add_dirt_to_same_position, perf_count_clean_with_penalty)
    env.add_agent(SimpleReflexAgent(), Position(0, 0))
    print env
    env.run(1000, verbose=False)
