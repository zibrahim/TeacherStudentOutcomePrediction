import re
from bisect import bisect_left
import datetime as dt


def getDayWrapper(some_timedelta):
    return getDay(str(some_timedelta))
def getHourWrapper(some_timedelta):
    return getHour(str(some_timedelta))

def binSearch(a, x):
   i = bisect_left(a, x)
   if i != len(a) and a[i] == x:
      return i
   else:
      return -1

#Get components from a string object corresponding to a timedelta, in the format: x days, HH:MM:SS
def getDay( ox ):
    day_value = "0"
    all_tokens = re.split('[a-z:, ]', ox)
    all_tokens = [token for token in all_tokens if token != '']

    if len(all_tokens) ==4:
        day_value = all_tokens[0]
    return day_value

def getHour( ox ):
    all_tokens =re.split('[a-z:, ]', ox)
    all_tokens = [token for token in all_tokens if token !='']

    if len(all_tokens) == 4:
        return all_tokens[1]
    else:
        return all_tokens[0]


def getMinute( ox ):
   all_tokens = re.split('[a-z:, ]', ox)
   all_tokens = [token for token in all_tokens if token != '']

   if len(all_tokens) == 4 :
       return all_tokens[2]
   else :
       return all_tokens[1]


class Timer():

	def __init__(self):
		self.start_dt = None

	def start(self):
		self.start_dt = dt.datetime.now()

	def stop(self):
		end_dt = dt.datetime.now()
		print('Time taken: %s' % (end_dt - self.start_dt))