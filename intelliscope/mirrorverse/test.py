from multiprocessing import Process, Queue
import numpy as np
import click
import json
from tqdm import tqdm
from time import time
from collections import Counter, defaultdict


class ComputeContainer(object):

    def __init__(self, agents, sub_queue, pub_queue):
        self.pub_queue = pub_queue
        self.sub_queue = sub_queue
        self.agents = agents
        self.read_state = None
        self.write_state = []

    def sense(self):
        self.read_state = self.sub_queue.get()

    def act(self):
        for agent in self.agents:
            agent.sense(self.read_state)
            agent.act()

    def signal(self):
        self.write_state = []
        for agent in self.agents:
            self.write_state.append(agent.signal())
        self.read_state = None

        self.pub_queue.put(self.write_state)


def job(agents, trigger_queue, sub_queue, pub_queue):
    container = ComputeContainer(agents, sub_queue, pub_queue)
    container.signal()
    while True:
        trigger = trigger_queue.get()
        if not trigger:
            break
        container.sense()
        container.act()
        container.signal()
    print('process finished')


class Exchanger(object):
    def __init__(self, sub_queues, pub_queues):
        self.sub_queues = sub_queues
        self.pub_queues = pub_queues
        self.state = None

    def accumulate_state(self):
        self.state = {
            'tags': [],
            'positions': [],
            'weights': [],
            'winners': defaultdict(int)
        }
        for pub_queue in self.pub_queues:
            container_states = pub_queue.get()
            for new_state in container_states:
                if new_state[0] == 'area':
                    for tag, won_value in new_state[1].items():
                        self.state['winners'][tag] += won_value
                else:
                    tag, position, weight = new_state[1]
                    self.state['tags'].append(tag)
                    self.state['positions'].append(position)
                    self.state['weights'].append(weight)

        self.state['tags'] = np.array(self.state['tags'])
        self.state['positions'] = np.array(self.state['positions'])
        self.state['weights'] = np.array(self.state['weights'])

    def exchange(self):
        self.accumulate_state()
        for sub_queue in self.sub_queues:
            sub_queue.put(self.state)
        self.state = None


class Area(object):
    def __init__(self, value, centroid, unit_territory_radius, unit_value):
        self.value = np.floor(value / unit_value) * unit_value
        self.units = int(self.value / unit_value)
        self.centroid = centroid
        self.unit_territory_radius = unit_territory_radius
        self.unit_value = unit_value
        self.wins = {}

    def sense(self, state):
        self.fish_positions = state['positions']
        self.fish_weights = state['weights']
        self.fish_tags = state['tags']
    
    def act(self):
        differences = self.fish_positions - self.centroid
        distances = np.linalg.norm(differences, axis=1)
        territory_sizes = self.unit_territory_radius * np.sqrt(self.fish_weights)

        probs = distances
        probs[probs < territory_sizes] = 0
        probs = probs * self.fish_weights
        probs = probs / np.sum(probs)

        winners = np.random.choice(
            self.fish_tags, size=self.units, p=probs, replace=True
        )

        self.wins = {}
        for tag, num_wins in Counter(winners).items():
            self.wins[tag] = num_wins * self.unit_value

    def signal(self):
        return 'area', self.wins


class Fish(object):
    def __init__(
        self, tag, position, weight, habitat_centroids, habitat_values, 
        unit_territory_radius, rate_limiter
    ):
        self.tag = tag
        self.position = position
        self.weight = weight
        self.habitat_centroids = habitat_centroids
        self.habitat_values = habitat_values
        self.unit_territory_radius = unit_territory_radius
        self.territory_radius = unit_territory_radius * np.sqrt(self.weight)
        self.rate_limiter = rate_limiter

    def sense(self, state):
        self.fish_weights = np.array(state['weights'])
        self.fish_positions = np.array(state['positions'])
        self.fish_tags = np.array(state['tags'])

    def act(self):
        differences = self.fish_positions - self.position
        distances = np.linalg.norm(differences, axis=1)
        territory_sizes = self.unit_territory_radius * np.sqrt(self.fish_weights)
        interaction_distances = self.territory_radius + territory_sizes

        # filter down to observable fish
        weights = self.fish_weights[(distances <= interaction_distances) & (self.fish_tags != self.tag)]
        positions = self.fish_positions[(distances <= interaction_distances) & (self.fish_tags != self.tag)]
        tags = self.fish_tags[(distances <= interaction_distances) & (self.fish_tags != self.tag)]
        territory_sizes = self.unit_territory_radius * np.sqrt(weights)

        # filter down to observable habitats
        differences = self.habitat_centroids - self.position
        habitat_distances = np.linalg.norm(differences, axis=1)
        habitat_centroids = self.habitat_centroids[habitat_distances <= self.territory_radius]
        habitat_values = self.habitat_values[habitat_distances <= self.territory_radius]

        # compute expected value of each habitat
        best_value = -float('inf')
        best_centroid = None
        for centroid, value in zip(habitat_centroids, habitat_values):
            differences = positions - centroid
            distances = np.linalg.norm(differences, axis=1)
            sub_weights = weights[distances <= territory_sizes]
            expected_value = self.weight / (np.sum(sub_weights) + self.weight) * value
            if expected_value > best_value:
                best_value = expected_value
                best_centroid = centroid

        direction = (best_centroid - self.position)
        direction = direction / np.linalg.norm(direction)
        self.position = self.position + direction * self.territory_radius * self.rate_limiter

    def signal(self):
        return 'fish', (self.tag, self.position, self.weight)

def divide_agents(agents, num_processes):
    divided_agents = [list() for _ in range(num_processes)]
    for i, agent in enumerate(agents):
        divided_agents[i % num_processes].append(agent)
    return divided_agents

@click.command()
@click.option('-n', '--num-agents', default=100, type=int)
@click.option('-p', '--num-processes', default=10, type=int)
@click.option('-s', '--time-steps', default=100, type=int)
@click.option('-o', '--output-file', default='output.csv')
def main(
    num_agents, num_processes, time_steps, output_file
):
    pub_queues = [Queue() for _ in range(num_processes)]
    sub_queues = [Queue() for _ in range(num_processes)]

    print('initializing habitats')
    stream_bounds = [0, 1]
    habitat_radius = 0.01

    habitat_centroids = np.arange(
        stream_bounds[0] + habitat_radius, 
        stream_bounds[1],
        habitat_radius * 2
    )

    habitat_values = habitat_centroids
    habitat_centroids = np.array([
        [x] for x in habitat_centroids
    ])
    unit_value = np.min(habitat_values)
    unit_territory_radius = (habitat_radius * 2) * 3

    habitats = [
        Area(value, centroid, unit_territory_radius, unit_value)
        for value, centroid in zip(habitat_values, habitat_centroids)
    ]
    print(len(habitats), 'habitats created')

    print('initializing fish')
    weight_bounds = [
        max((habitat_radius / unit_territory_radius) ** 2, 0.25), 1
    ]
    rate_limiter = 0.25
    fish = [
        Fish(
            tag, np.array([np.random.uniform(*stream_bounds)]), np.random.uniform(*weight_bounds), 
            habitat_centroids, habitat_values, unit_territory_radius, rate_limiter
        )
        for tag in range(num_agents)
    ]

    print('starting processes')
    agents = habitats + fish
    divided_agents = divide_agents(agents, num_processes)
    trigger_queues = [Queue() for _ in range(num_processes)]
    processes = [
        Process(target=job, args=(sub_agents, trigger_queue, sub_queue, pub_queue))
        for sub_agents, trigger_queue, sub_queue, pub_queue in zip(divided_agents, trigger_queues, sub_queues, pub_queues)
    ]
    for process in processes:
        process.start()

    print('building exchanger')
    exchanger = Exchanger(sub_queues, pub_queues)

    print('initializing state')
    exchanger.exchange()
    
    print('running simulation')
    for _ in tqdm(range(time_steps)):
        for trigger_queue in trigger_queues:
            trigger_queue.put(1)
        exchanger.exchange()

    print('sending stop signals')
    for trigger_queue in trigger_queues:
        trigger_queue.put(0)

    print('waiting for processes to finish')
    for process in processes:
        process.join()

    print('clearing sub queues')
    while not all(queue.empty() for queue in sub_queues):
        for queue in sub_queues:
            while not queue.empty():
                queue.get()

    print('writing results')
    results = [list(agent.position) for agent in fish]
    with open(output_file, 'w') as f:
        json.dump(results, f)
    print('done')

if __name__ == '__main__':
    main()