import datetime as dt
from dateutil import rrule 

def trading_holidays(start = dt.date.today() - dt.timedelta(days = 365), end = dt.date.today()):
    rs = rrule.rruleset()                                                                                                    # Create rule
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart = start, until = end, bymonth = 12, bymonthday = 31, byweekday = rrule.FR))   # New Years Day  
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart = start, until = end, bymonth = 1, bymonthday = 1))                           # New Years Day  
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart = start, until = end, bymonth = 1, bymonthday = 2, byweekday = rrule.MO))     # New Years Day    
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart = start, until = end, bymonth = 1, byweekday = rrule.MO(3)))                  # Martin Luther King Day   
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart = start, until = end, bymonth = 2, byweekday = rrule.MO(3)))                  # Washington's Birthday
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart = start, until = end, byeaster = -2))                                         # Good Friday
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart = start, until = end, bymonth = 5, byweekday = rrule.MO(-1)))                 # Memorial Day
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart = start, until = end, bymonth = 7, bymonthday = 3, byweekday = rrule.FR))     # Independence Day
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart = start, until = end, bymonth = 7, bymonthday = 4))                           # Independence Day
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart = start, until = end, bymonth = 7, bymonthday = 5, byweekday = rrule.MO))     # Independence Day
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart = start, until = end, bymonth = 9, byweekday = rrule.MO(1)))                  # Labor Day
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart = start, until = end, bymonth = 11, byweekday = rrule.TH(4)))                 # Thanksgiving Day
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart = start, until = end, bymonth = 12, bymonthday = 24, byweekday = rrule.FR))   # Christmas  
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart = start, until = end, bymonth = 12, bymonthday = 25))                         # Christmas  
    rs.rrule(rrule.rrule(rrule.YEARLY, dtstart = start, until = end, bymonth = 12, bymonthday = 26, byweekday = rrule.MO))   # Christmas 
    rs.exrule(rrule.rrule(rrule.WEEKLY, dtstart = start, until = end, byweekday = (rrule.SA, rrule.SU)))                     # Exclude holidays on weekends 
    return rs

trading_holidays()