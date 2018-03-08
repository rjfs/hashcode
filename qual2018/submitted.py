"""
This is the code that was submitted to the platform during Online Qualification Round
Total Score: 17,712,424
"""
from joblib import Parallel, delayed
import multiprocessing


def parallel():
    num_cores = multiprocessing.cpu_count()
    inputs = ['a_example', 'b_should_be_easy', 'c_no_hurry', 'd_metropolis', 'e_high_bonus']
    Parallel(n_jobs=num_cores)(delayed(simulation)(i) for i in inputs)


def simulation(f):
    sim = Simulation(f)
    sim.run()


def main():
    for f in ['b_should_be_easy']:
        sim = Simulation(f)
        sim.run()


class Simulation:

    def __init__(self, name):
        self.name = name
        self.inputs = get_obj(name + '.in')
        self.fleet = [Vehicle() for _ in range(self.inputs.vehicles)]

    @property
    def T(self):
        return self.inputs.steps

    @property
    def bonus(self):
        return self.inputs.bonus

    def run(self):
        rides = self.inputs.rides[:]
        for dt in range(self.T):
            self.update_vehicles_states(dt)
            for iv, vehicle in enumerate(self.fleet):
                if not vehicle.in_ride:
                    pr = vehicle.possible_rides(dt=0, rides=rides, bonus=self.bonus, sim_time=self.T)
                    selected = self.select_ride(pr) if len(pr) > 0 else None

                    if selected is not None:
                        # print('Assigning ride %d to vehicle %d' % (selected.ride_n, iv))
                        vehicle.add_ride(dt=dt, ride=selected)
                        rides.remove(selected)

        self.generate_solution()

    def select_ride(self, pr):
        # max_score = max([i[1] for i in pr])
        # Update scores
        # pr = [(r, self.score_ride(r, max_score, s)) for r, s in pr]
        sort_pr = sorted(pr, key=lambda x: x[1], reverse=True)

        return sort_pr[0][0]

    def score_ride(self, ride, max_score, s):
        a = 5.0
        b = -1.0
        return a * float(s) / max_score + b * float(ride.earliest_start) / self.T

    def generate_solution(self):
        fname = self.name + '.out'
        f = open(fname, 'w')
        for i, v in enumerate(self.fleet):
            rides_numbers = ' '.join([str(r.ride_n) for r in v.rides])
            f.write('%s %s\n' % (len(v.rides), rides_numbers))

        print('Written to %s' % fname)

    def update_vehicles_states(self, dt):
        for v in self.fleet:
            v.update_state(dt)


class Inputs:

    def __init__(self, header, rides):
        header_lst = header.split(' ')
        self.rows = int(header_lst[0])
        self.columns = int(header_lst[1])
        self.vehicles = int(header_lst[2])
        self.n_rides = int(header_lst[3])
        self.bonus = int(header_lst[4])
        self.steps = int(header_lst[5])
        self.rides = get_rides_objects(rides)


class Vehicle:

    def __init__(self):
        self.x = 0
        self.y = 0
        self.ride_finnish = None
        self.current_ride = None
        self.rides = []

    @property
    def in_ride(self):
        return self.ride_finnish is not None

    def possible_rides(self, dt, rides, bonus, sim_time):
        """

        :param dt:
        :param rides:
        :return: list of tuples
            [(Ride, score)]
        """
        out = []
        for r in rides:
            score = self.get_score(dt, r, bonus=bonus, sim_time=sim_time)
            if score > 0:
                out.append((r, score))

        return out

    def update_state(self, dt):
        if self.ride_finnish is not None:
            if dt >= self.ride_finnish:
                self.ride_finnish = None
                self.x = self.current_ride.final_x
                self.y = self.current_ride.final_y
                self.current_ride = None

    def add_ride(self, dt, ride):
        self.current_ride = ride
        self.rides.append(ride)
        dvs = abs(self.x - ride.initial_x) + abs(self.y - ride.initial_y)
        waiting_time = max(0, ride.earliest_start - dvs - dt)
        self.ride_finnish = dt + dvs + waiting_time + ride.distance()

    def get_score(self, dt, ride, bonus, sim_time):
        # Distance vehicle - start
        score = 0
        d_vs = abs(self.x - ride.initial_x) + abs(self.y - ride.initial_y)
        waiting_time = max(0, ride.earliest_start - d_vs - dt)
        if dt + d_vs + waiting_time + ride.distance() <= ride.latest_finnish:
            score = ride.distance()
            if dt + d_vs <= ride.earliest_start:
                score += bonus

            if waiting_time + d_vs > int(0.5 * sim_time):
                score = 1

        return score

    def is_possible(self, dt, ride):
        """
        Return 'possible' if possible, 'bonus' if possible with bonus, None if not possible
        :param ride:
        :return:
        """
        # Distance vehicle - start
        d_vs = abs(self.x - ride.initial_x) + abs(self.y - ride.initial_y)
        if dt + d_vs + ride.distance() <= ride.latest_finnish:
            if dt + d_vs <= ride.earliest_start:
                return 'bonus'
            else:
                return 'possible'


def get_rides_objects(rides):
    rides_mat = [list(line[:-1].split(' ')) for line in rides]
    return [Ride(ride_n, r) for ride_n, r in enumerate(rides_mat)]


class Ride:

    def __init__(self, ride_n, ride_array):
        self.ride_n = ride_n
        self.initial_x = int(ride_array[0])
        self.initial_y = int(ride_array[1])
        self.final_x = int(ride_array[2])
        self.final_y = int(ride_array[3])
        self.earliest_start = int(ride_array[4])
        self.latest_finnish = int(ride_array[5])

    def distance(self):
        return abs(self.final_x - self.initial_x) + abs(self.final_y - self.initial_y)


def get_obj(fname):
    header, rides = read_file(fname)
    return Inputs(header, rides)


def read_file(fname):
    f = open(fname, 'r')
    return f.readline(), f.readlines()


if __name__ == '__main__':
    parallel()
    # main()
