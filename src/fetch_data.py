# Run this file to install all of the 2026 regular season statcast data into the data/raw folder

import pybaseball as pyb

pyb.cache.enable()
month_dates = [['2026-03-01', '2026-03-31'],
               ['2026-04-01', '2026-04-30'],
               ['2026-05-01', '2026-05-31'],
               ['2026-06-01', '2026-06-30'],
               ['2026-07-01', '2026-07-31'],
               ['2026-08-01', '2026-08-31'],
               ['2026-09-01', '2026-10-31']]

for start_dt, end_dt in month_dates:
    data = pyb.statcast(start_dt=start_dt, end_dt=end_dt)
    # Filter for regular season games only
    data = data[data["game_type"] == "R"]
    data.to_csv(f'data/raw/statcast_{start_dt[0:4]}_{start_dt[5:7]}.csv', index=False)


