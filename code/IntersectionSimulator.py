# standard libraries
import os
import fileinput
import sys
import re

path = "/home/lucianodp/Desktop/traffic_control"  # parent folder
os.environ['CLASSPATH'] = path + "/jar/beats2.jar"  # set path to jar file

# third-party libraries
import jnius  # Python-Java API - ALWAYS IMPORT IT AFTER SETTING CLASSPATH
import numpy as np

# properties file
intersection_name = 'intersection'
prop_path = path + '/data/properties/'+intersection_name+'.properties'
xml_path = path + '/data/config/'+intersection_name+'.xml'

# Java DataTypes - for use with HashMap
jMap = jnius.autoclass('java.util.HashMap')
jInt = jnius.autoclass('java.lang.Integer')
jLong = jnius.autoclass('java.lang.Long')
jFloat = jnius.autoclass('java.lang.Float')
jDouble = jnius.autoclass('java.lang.Double')
LinkData = jnius.autoclass('api.data.LinkData')


# beats2 java simulator wrapper
class IntersectionSimulator(object):
    def __init__(self, sources_id, eff_cycle_time):
        # attributes
        self.n_sources = len(sources_id)
        self.sources_id = sources_id

        self.eff_cycle_length = eff_cycle_time  # duration of each cycle (discounted yellow and all-red times)

        self.beats = jnius.autoclass('runner.BeATS')  # import BeATS class through jnius

        # load properties files (xml and other parameters)
        self.beats.load_and_check(prop_path)

    def initialize(self):
        """
        Initializes beats simulator, setting occupations to 0. It also sets the green times
        and occupations for the next run.

        :return: initial occupations (zero)
        """
        self.beats.api.initialize()

    def run(self, action, occupations=None):
        """
        This function sets the links occupations and the green times, and then runs a simulation for a given duration.
        The duration is specified in the properties file.

        :occupations: number of cars in each link
        :action: proportion of green time to each phase
        """
        self.set_green_time(action)  # set green times

        self.beats.api.initialize()  # initialize must be called to SET THE GREEN TIMES for the next run

        if occupations is not None:
            self.set_occupation(occupations)  # set occupations

        self.beats.api.run()  # run simulation

    # GET METHODS
    def get_queue(self):
        """
        Returns the queue length of each lane group right before the last time the signal changed to green.

        :return: queue lengths for each lane group
        """
        # get signal data
        signals = self.beats.api.get_signal_data()

        # get single signal
        signal = signals.toArray()[0]

        # find lane groups occupations
        lanegroups = signal.lanegroup_queueStartingGreen.values()

        # iterate get values
        queue = []
        it = lanegroups.iterator()

        while it.hasNext():
            queue.append(it.next())

        # convert to array
        return np.array(queue)

    def get_occupation(self):
        """
        Returns the current occupation of each source link (passed as parameters to our class).
        Sink links are usually empty, so we do not take them into account.

        :return: occupations of source links, as a numpy array.
        """
        occupation = np.empty(self.n_sources)

        # get link data
        data = self.beats.api.get_link_data()

        link_iter = data.iterator()  # iterator
        while link_iter.hasNext():
            # current link
            d = link_iter.next()

            if d.link_id in self.sources_id:  # source node
                pos = id_to_pos(d.link_id)
                occupation[pos] = d.vehicles

        return occupation

    # SET METHODS
    def set_occupation(self, occupation):
        """
        Set the occupations for the next run. Fractional values are truncated and negative values set to 0.
        :param occupation: numpy array containing the desired occupations of each source link
        """
        assert occupation.shape == (self.n_sources,), "Wrong occupation format"

        for link_id, occup in zip(self.sources_id, occupation):
            data = LinkData(jLong(link_id), jDouble(occup))  # create LinkData. Only Java formats are accepted
            self.beats.api.set_link_data(data)

    def set_green_time(self, action):
        """
        Set green times of each phase.

        :param action: array containing the proportion of green time for each phase
        """

        controller_id = 1  # this value is set in the intersection's XML file
        assert action.shape == (self.n_sources,) and abs(np.sum(action) - 1.0) < 0.001, \
            "Invalid green times: {}".format(action)

        action = action.clip(0.0001, 0.9999)  # clip values - 0 and 1 are not allowed!

        greens = jMap()
        for pos, prop in enumerate(action):
            # find values
            green_time = prop * self.eff_cycle_length

            # append to HashMap
	    link_id = pos_to_id(pos)
            greens.put(jInt(link_id), jFloat(green_time))

        # set green times
        self.beats.api.set_green_time(controller_id, greens)

    def set_demands(self, demands):
        """
        Rewrites the xml scenario file, replacing the demands as they are found. File is them reloaded.
        :param demands: list of new demands.
        :return:
        """
        assert len(demands) == self.n_sources, 'Wrong number of demands!'

        i = 0
        str_demands = ['>' + str(d) + '<' for d in demands]

        for line in fileinput.input(xml_path, inplace=True):
            if '</demand>' in line:
                line = re.sub('>(\d+)<', str_demands[i], line)
                i += 1
            sys.stdout.write(line)

        # reload properties file
        self.beats.load_and_check(prop_path)


# AUXILIARY FUNCTIONS
def id_to_pos(ide):
    return ide - 1


def pos_to_id(pos):
    return pos + 1

