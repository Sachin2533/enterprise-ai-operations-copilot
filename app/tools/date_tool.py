from datetime import datetime


def get_current_date():
    """
    Returns current date
    """

    return datetime.now().strftime(
        "%d-%m-%Y"
    )


def get_current_time():
    """
    Returns current time
    """

    return datetime.now().strftime(
        "%H:%M:%S"
    )


def get_day_of_week():
    """
    Returns current day
    """

    return datetime.now().strftime(
        "%A"
    )