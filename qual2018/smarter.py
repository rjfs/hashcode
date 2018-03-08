"""
In this file, a smarter implementation is done in terms of algorithm and also running time performance.

Major changes:
    1) Rides are now sorted by maximum initial time. This makes the removal of expired rides more efficient.
    2) Distances from vehicle to all the possible rides computations are now vectorized, using numpy, what makes the
    code run more efficiently.
    3) A smarter algorithm is implemented. When computing the best ride for a vehicle, a ride 1 is better than a ride 2
     in case it is better to do ride 1 followed by ride 2 than doing ride 2 followed by ride 1. Besides, only the rides
     that are close enough to the vehicle are considered.

Total Score (in extended round):
    A: 10
    B: 176,877
    C: 15,716,651
    D: 8,416,836
    E: 21,428,945
    Total: 45,739,319

Note: The results for each file might have been generated with different simulation parameters, for example, the
percentage parameter used for rides splitting.

"""
import joblib
import multiprocessing
import time
import numpy as np
import random


def parallel():
    """ Run each file in a different process """
    inputs = ['a_example', 'b_should_be_easy', 'c_no_hurry', 'd_metropolis', 'e_high_bonus']
    n_proc = min(len(inputs), multiprocessing.cpu_count())
    joblib.Parallel(n_jobs=n_proc)(joblib.delayed(run_simulation)(i) for i in inputs)


def run_simulation(f):
    """ Run simulation for given file name """
    sim = Simulation(f)
    sim.run()


def main():
    # for f in ['a_example', 'b_should_be_easy', 'c_no_hurry', 'd_metropolis', 'e_high_bonus']:
    for f in ['e_high_bonus']:
        run_simulation(f)


class Simulation:

    def __init__(self, name):
        self.name = name
        self.out_suffix = '_sm'
        self.inputs = get_obj(name + '.in')
        self.fleet = [Vehicle() for _ in range(self.inputs.vehicles)]

    @property
    def T(self):
        # Number of simulation steps
        return self.inputs.steps

    @property
    def bonus(self):
        # Bonus given by a ride that starts on time
        return self.inputs.bonus

    def run(self):
        """ Run simulation and write solution to a file """
        # Get list of rides sorted by maximum initial time
        rides = sorted(self.inputs.rides[:], key=lambda x: x.max_initial_time)
        for dt in range(self.T):
            # Remove expired rides
            self.remove_expired_rides(dt, rides)
            # Update vehicles states
            self.update_vehicles_states(dt)
            free_vehicles = [v for v in self.fleet if not v.in_ride]
            if dt % 1000 == 0:
                print('[%i/%i] Rides Left: %d' % (dt, self.T, len(rides)))
            for vehicle in free_vehicles:
                if len(rides) > 0:
                    selected = vehicle.select_ride(dt=dt, rides=rides, bonus=self.bonus)
                    if selected is not None:
                        vehicle.add_ride(dt=dt, ride=selected)
                        rides.remove(selected)

        # Generate output file
        self.generate_solution()

    def remove_expired_rides(self, dt, rides):
        """
        Removes rides that can't be finished anymore
        Assumes rides are sorted by maximum initial time
        """
        i = 0
        for r in rides:
            if r.max_initial_time < dt:
                i += 1
            else:
                break

        # Delete first i elements
        del rides[:i]

    def generate_solution(self):
        """ Generates output file """
        fname = self.name + self.out_suffix + '.out'
        f = open(fname, 'w')
        for i, v in enumerate(self.fleet):
            rides_numbers = ' '.join([str(r.ride_n) for r in v.rides])
            f.write('%s %s\n' % (len(v.rides), rides_numbers))

        print('Written to %s' % fname)

    def update_vehicles_states(self, dt):
        """ Updates states for each vehicle in fleet """
        for v in self.fleet:
            v.update_state(dt)


class Inputs:

    def __init__(self, header, rides):
        """ Input file parameters """
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
        self.ride_finish = None
        self.current_ride = None
        self.rides = []

    @property
    def in_ride(self):
        """ Returns True if vehicle is in a ride """
        return self.ride_finish is not None

    def get_distances(self, rides):
        """
        Compute distances between vehicle and each one of given rides
        :param rides: list of Ride
            List of rides
        :return: list of tuples
            (ride, distance) where distance is the distance between current vehicle position and ride starting point.
        """
        rides_x = np.array([r.initial_x for r in rides])
        rides_y = np.array([r.initial_y for r in rides])
        distances = np.abs(rides_x - self.x) + np.abs(rides_y - self.y)
        return zip(rides, distances)

    def get_couple_score(self, r1, r2, dt, d_r1, bonus):
        """
        Get score of doing ride r1 and ride r2 right after.
        The score is the ride distance + bonus in case the ride can start on time.
        """
        # Get ride score
        waiting_time = max(0, r1.earliest_start - d_r1 - dt)
        ride_start = dt + d_r1 + waiting_time
        ride_finish = ride_start + r1.distance
        score = 0
        if ride_finish <= r1.latest_finish:
            score = r1.distance
            if dt + d_r1 <= r1.earliest_start:
                score += bonus

            # Compute distance between ride1 finish and ride2 starting point
            d_rides = abs(r2.initial_x - r1.final_x) + abs(r2.initial_y - r1.final_y)
            if ride_finish + d_rides + r2.distance <= r2.latest_finish:
                # Add second ride distance
                score += r2.distance
                if ride_finish + d_rides <= r2.earliest_start:
                    # Add second ride bonus
                    score += bonus

        return score

    def random_split_rides(self, rides_distances):
        """
        Splits rides into nearby rides and far away rides.
        The split point is randomly computed.
        :param rides_distances: list of tuples
            (ride, distance) where distance is the distance between current vehicle position and ride starting point.
        :return: tuple of lists
            First element is nearby rides and second is far away rides
        """
        # Sort rides by distance
        rides_distances.sort(key=lambda x: x[1])
        # Get random split point
        split_point = random.randint(1, len(rides_distances))
        return rides_distances[:split_point], rides_distances[split_point:]

    def pct_split_rides(self, rides_distances, pct=0.4):
        """
        Splits rides into nearby rides and far away rides.
        The split point is given by inputed percentage.
        :param rides_distances: list of tuples
            (ride, distance) where distance is the distance between current vehicle position and ride starting point.
        :param pct: float, default 0.1
            Split point percentage
        :return: tuple of lists
            First element is nearby rides and second is far away rides
        """
        # Sort rides by distance
        rides_distances.sort(key=lambda x: x[1])
        # Get random split point
        split_point = int(len(rides_distances) * pct)
        return rides_distances[:split_point], rides_distances[split_point:]

    def select_ride(self, dt, rides, bonus):
        """ Selects best ride for vehicle """
        # Get distances between vehicle and starting point of each ride
        distances = self.get_distances(rides)
        # Find best ride for vehicle
        near_rides, far_rides = self.random_split_rides(distances)
        best = self.get_best_ride(near_rides, dt=dt, bonus=bonus)
        if best is None:
            # If no ride was selected, search in far away rides
            best = self.get_best_ride(far_rides, dt=dt, bonus=bonus)

        return best

    def get_best_ride(self, rides, dt, bonus):
        """ Select best ride from given list of rides """
        best = None
        dist_best = None
        for r, d_vr in rides:
            if best is None:
                best = r
                dist_best = d_vr
            else:
                new_score = self.get_couple_score(r1=r, r2=best, dt=dt, d_r1=d_vr, bonus=bonus)
                # Get score for doing best ride before ride
                prev_score = self.get_couple_score(r1=best, r2=r, dt=dt, d_r1=dist_best, bonus=bonus)

                if new_score > prev_score:
                    # print('Ride %d: dist=%d (%d -> %d)' % (r.ride_n, r.distance, r.earliest_start, r.latest_finish))
                    best = r
                    dist_best = d_vr

        return best

    def update_state(self, dt):
        """ Updates vehicle state """
        if self.ride_finish is not None:
            if dt >= self.ride_finish:
                self.ride_finish = None
                self.x = self.current_ride.final_x
                self.y = self.current_ride.final_y
                self.current_ride = None

    def add_ride(self, dt, ride):
        """ Assigns given ride to vehicle """
        self.current_ride = ride
        self.rides.append(ride)
        dvs = abs(self.x - ride.initial_x) + abs(self.y - ride.initial_y)
        waiting_time = max(0, ride.earliest_start - dvs - dt)
        self.ride_finish = dt + dvs + waiting_time + ride.distance


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
        self.latest_finish = int(ride_array[5])
        self.distance = self.compute_distance()
        self.max_initial_time = self.latest_finish - self.distance

    def compute_distance(self):
        """ Computes ride distance """
        return abs(self.final_x - self.initial_x) + abs(self.final_y - self.initial_y)


def get_obj(fname):
    f = open(fname, 'r')
    header, rides = f.readline(), f.readlines()
    return Inputs(header, rides)


def profile():
    import cProfile
    cProfile.run('main()', sort='cumulative')


if __name__ == '__main__':
    t0 = time.time()
    parallel()
    # main()
    # profile()
    print('Ran in %.2f seconds' % (time.time() - t0))
