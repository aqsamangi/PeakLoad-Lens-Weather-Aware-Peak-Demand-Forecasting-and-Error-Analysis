# Data

The raw dataset is **not** included in this repository.

1. Download **Power Consumption of Tetouan City** from the UCI Machine Learning Repository:
   <https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city>
2. Unzip it and place the file in this folder with the original name:

   ```
   data/Tetuan City power consumption.csv
   ```

3. Run from the project root: `python src/data_audit.py`

To keep the file elsewhere, pass `--data /path/to/file.csv` to the scripts or set the environment variable `PEAKLOAD_DATA`.

License: CC BY 4.0. Citation: Salam, A. & El Hibaoui, A. (2018). *Power Consumption of Tetouan City* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5B034
