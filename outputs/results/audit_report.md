# Data audit report

- Rows found: **52,416** (UCI page lists 52,417; match: False). Columns: 9 (expected 9).
- Timestamps: 2017-01-01 00:00:00 to 2017-12-30 23:50:00; unparsed: 0; chronological: True.
- Duplicate rows: 0; duplicate timestamps: 0.
- Interval counts: {'0 days 00:10:00': 52415}; missing 10-minute grid slots: 0.
- Calendar dates absent from the year: ['2017-12-31'].
- Missing values: 0; non-numeric: 0; non-finite: 0.
- Extreme values (beyond 3xIQR fences, reported only): {'Temperature': 0, 'Humidity': 0, 'Wind Speed': 0, 'general diffuse flows': 0, 'diffuse flows': 1690, 'Zone 1 Power Consumption': 0, 'Zone 2  Power Consumption': 0, 'Zone 3  Power Consumption': 18}.
- Longest run of identical consumption values: {'Zone 1 Power Consumption': 3, 'Zone 2  Power Consumption': 3, 'Zone 3  Power Consumption': 3}.
- Zone 1 10-minute absolute change: {'median': 310.3, 'p99': 2480.2, 'max': 19344.7}.
- Observations per month: {'2017-01': 4464, '2017-02': 4032, '2017-03': 4464, '2017-04': 4320, '2017-05': 4464, '2017-06': 4320, '2017-07': 4464, '2017-08': 4464, '2017-09': 4320, '2017-10': 4464, '2017-11': 4320, '2017-12': 4320}.
