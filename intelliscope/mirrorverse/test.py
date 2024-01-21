from multiprocessing import Process, Queue
import numpy as np
import click
import json
from tqdm import tqdm
from time import time

class Agent(object):
    def __init__(self, position, min_difference, time_delta, velocity, friction, pub_queue, sub_queue):
        self.pub_queue = pub_queue
        self.sub_queue = sub_queue
        self.position = position
        self.min_difference = min_difference
        self.time_delta = time_delta
        self.velocity = velocity
        self.friction = friction
        self.positions = []

    def sense(self):
        self.positions = []
        while not self.sub_queue.empty():
            signal = self.sub_queue.get()
            self.positions.append(signal)

    def act(self):
        differences = self.position - np.array(self.positions)
        differences[differences == 0] = np.random.choice([-1, 1], differences[differences == 0].shape) * self.min_difference
        forces = 1 / (differences ** 2) 
        forces[forces > 1 / (self.min_difference ** 2)] = 1 / (self.min_difference ** 2)
        forces = forces * np.sign(differences)
        force = np.sum(forces)
        self.velocity += force * self.time_delta
        # now apply friction
        friction_force = self.friction * self.velocity
        current_sign = np.sign(self.velocity)
        self.velocity -= friction_force * self.time_delta
        if np.sign(self.velocity) != current_sign:
            self.velocity = 0

        self.position += self.velocity * self.time_delta

    def signal(self):
        self.pub_queue.put(self.position)

def divide_agents(agents, num_processes):
    divided_agents = [list() for _ in range(num_processes)]
    for i, agent in enumerate(agents):
        divided_agents[i % num_processes].append(agent)
    return divided_agents

def do_exchange(pub_queues, sub_queues):
    for input_queue in pub_queues:
        input_ = input_queue.get()
        for output_queue in sub_queues:
            output_queue.put(input_)

def job(agents, trigger_queue, finished_queue):
    while True:
        trigger = trigger_queue.get()
        if not trigger:
            break
        for agent in agents:
            agent.sense()
            agent.act()
        for agent in agents:
            agent.signal()
        finished_queue.put(1)
    print('job done')

@click.command()
@click.option('-n', '--num-agents', default=25, type=int)
@click.option('-d', '--min-difference', default=0.5, type=float)
@click.option('-f', '--friction', default=0.05, type=float)
@click.option('-t', '--time-delta', default=0.1, type=float)
@click.option('-p', '--num-processes', default=8, type=int)
@click.option('-s', '--time-steps', default=100, type=int)
@click.option('-o', '--output-file', default='output.csv')
def main(
    num_agents, min_difference, friction, time_delta, num_processes, time_steps, output_file
):
    pub_queues = [Queue() for _ in range(num_agents)]
    sub_queues = [Queue() for _ in range(num_agents)]

    print('initializing agents')
    agents = [
        Agent(np.random.uniform(-1, 1), min_difference, time_delta, 0, friction, pub_queue, sub_queue)
        for pub_queue, sub_queue in zip(pub_queues, sub_queues)
    ]
    for agent in agents:
        agent.signal()
    do_exchange(pub_queues, sub_queues)

    print('starting processes')
    divided_agents = divide_agents(agents, num_processes)
    trigger_queues = [Queue() for _ in range(num_processes)]
    finished_queue = Queue()
    processes = [
        Process(target=job, args=(sub_agents, trigger_queue, finished_queue))
        for sub_agents, trigger_queue in zip(divided_agents, trigger_queues)
    ]
    for process in processes:
        process.start()
    
    print('running simulation')
    for _ in tqdm(range(time_steps)):
        for trigger_queue in trigger_queues:
            trigger_queue.put(1)
        for _ in range(num_processes):
            finished_queue.get()
        do_exchange(pub_queues, sub_queues)

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
    results = [agent.position for agent in agents]
    with open(output_file, 'w') as f:
        json.dump(results, f)
    print('done')

if __name__ == '__main__':
    main()