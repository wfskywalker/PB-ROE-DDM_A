import pandas as pd

from pbroe.history import backtest_signal_schedule


def test_backtest_schedule_uses_next_open_and_t_plus_21_open():
    dates = pd.bdate_range("2024-01-02", "2024-03-15")
    calendar = pd.DataFrame(
        {
            "cal_date": dates.strftime("%Y%m%d").astype(int),
            "is_open": 1,
        }
    )
    schedule = backtest_signal_schedule(calendar, 20240101, 20240229)
    january = schedule.set_index("signal_date").loc[20240131]
    opened = calendar.cal_date.tolist()
    index = opened.index(20240131)
    assert january.entry_date == opened[index + 1]
    assert january.label_exit_date == opened[index + 21]
    assert january.next_entry_date == 20240301
