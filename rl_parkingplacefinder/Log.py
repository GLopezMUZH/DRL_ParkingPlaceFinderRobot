import enum
import io
import sys

import time
from datetime import datetime

class Log():
    def print_elapsed_time(self,start_time,end_time):
        elapsed = end_time - start_time
        seconds = elapsed.days*86400 + elapsed.seconds # drop microseconds
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        print("Start: ", start_time)
        print("End: ", end_time)
        print("Elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}".format(**vars()))