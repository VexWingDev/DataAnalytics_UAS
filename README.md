# Exercise Project 1: Dataset Selection and Preliminary EDA

## Task 1: Selected datasets

### Regression dataset

- **Dataset:** CityFlow: Smart Urban Mobility and Traffic IoT
- **Source:** https://www.kaggle.com/datasets/mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot
- **Local file:** `data/cityflow/smart_city_traffic_mobility.csv`
- **Selected target:** `emission_estimate`
- **Dataset type:** Regression
- **Reason:** `emission_estimate` is a continuous numerical estimate of tailpipe emissions in kilograms of CO2.

### Important course restriction

The CityFlow data contains 204,000 hourly observations and a `timestamp` column. It is therefore a time-series dataset in addition to being usable for regression. The assignment says not to select a time-series dataset. This dataset is included here because it was explicitly requested, but instructor approval should be obtained before final submission. If the restriction is enforced, replace this dataset with a non-time-series regression dataset.

Other continuous targets in CityFlow include `traffic_flow_rate`, `congestion_score`, `fuel_waste_estimate`, and `average_wait_time`. Only one target should be selected for a regression report.

### Classification dataset

- **Dataset:** Student Social Media and Mental Health Impact
- **Source:** https://www.kaggle.com/datasets/shivasingh4945/student-social-media-and-mental-health-impact
- **Local file:** `data/student_mental_health/Student Social Media And Mental Health Impact.csv`
- **Selected target:** `Stress_Level`
- **Dataset type:** Classification
- **Reason:** `Stress_Level` is categorical, with the observed classes `Very High`, `High`, `Medium`, and `Low`.

The dataset contains demographic, social-media, lifestyle, and mental-health variables that can be used to investigate whether student stress category can be predicted from student behavior and characteristics.

## Task 2: Preliminary EDA findings to verify

The notebooks in this folder calculate the values below and create the required visuals. Running the final notebook cell generates the ydata-profiling HTML report when the profiling package is installed.

### CityFlow preliminary observations

- Shape: 204,000 rows and 47 columns.
- Missing values: 0 in the downloaded file.
- Complete duplicate rows: 0.
- The data includes a timestamp and repeated hourly observations, confirming the time-series concern.
- `emission_estimate` is continuous and is appropriate as a regression target if the dataset is approved.
- `congestion_level` is categorical and has four classes, but it is not the selected target for this report.
- Many zero values are structurally meaningful binary indicators, such as no accident, no nearby facility, or no event. They must not automatically be treated as missing values.
- ID-like columns and timestamp-derived columns require careful consideration because they may create leakage or violate the intended non-time-series task.

### Student dataset preliminary observations

- Shape: 5,000 rows and 13 columns.
- Missing values: 0 in the downloaded file.
- Complete duplicate rows: 2.
- `Stress_Level` has four observed categories: `Very High`, `High`, `Medium`, and `Low`.
- The target is not perfectly balanced. `Very High` is the largest class and `Low` is the smallest class, so class imbalance should be investigated before model training.
- Numeric lifestyle variables should be checked for unrealistic values, skewness, and outliers.
- Categorical columns should be checked for inconsistent spelling, capitalization, and high cardinality.

## How the preliminary EDA should be interpreted

This stage identifies possible data-quality and modeling issues. It does not fix them, except for necessary datatype conversion such as converting CityFlow `timestamp` to a datetime type. A profiling alert is a recommendation, not proof that a value is invalid. Each alert should be checked against the column's meaning and a conventional pandas or seaborn analysis.

The two notebooks include:

- datatype inspection
- shape and size discussion
- missing-value analysis
- duplicate-row analysis
- zero-value analysis
- invalid or suspicious-value checks
- target distribution and balance
- support-variable distributions
- high-cardinality checks
- preliminary issue lists

## AI-use disclosure section
