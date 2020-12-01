import enum

class GranularityLevel(enum.Enum):
    one_sec = 1
    thirty_sec = 2
    ten_min = 3,
    one_hour = 4,
    three_hour = 5,
    one_day = 6,
    one_week = 7,
    one_minute = 8,


class InputEnums(enum.Enum):
    lag_time_steps = 5
    lead_time_steps = 1
    input = ['Feature']
    confidence_interval_multiple_factor = 1
    test_train_split_size = 0.2