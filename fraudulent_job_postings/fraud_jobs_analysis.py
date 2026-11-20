"""Fraudulent job posting analysis with PySpark.

Compares fraudulent and genuine job advertisements: class balance, salary
extraction and statistics, salary distributions (boxplots and histograms),
two outlier-removal approaches, and the most frequent bi-grams and tri-grams
in the job descriptions.

Run: spark-submit fraud_jobs_analysis.py   (fraud_jobs.csv in the same folder)
Charts are saved as PNG files next to this script.
"""
import os
import re
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import matplotlib
import numpy as np
from pyspark.ml.feature import NGram, RegexTokenizer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import LongType

matplotlib.use("Agg")
import matplotlib.pyplot as plt

spark = SparkSession.builder.master("local[*]").appName("ECA_Q2").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Q2(a) Load the data and count fraudulent vs genuine postings
base = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(base, "outputs"), exist_ok=True)
df = spark.read.csv(os.path.join(base, "fraud_jobs.csv"),
                    header=True, inferSchema=True,
                    multiLine=True, quote='"', escape='"')
print("Schema of the dataset:")
df.printSchema()

total_ads = df.count()
print("Total number of job advertisements:", total_ads)

df = df.withColumn("job_type",
                   F.when(F.col("fraudulent") == 1, "Fraudulent").otherwise("Genuine"))
(df.groupBy("job_type").count()
 .withColumn("percentage", F.round(F.col("count") / total_ads * 100, 2))
 .orderBy("job_type")
 .show())


# Q2(b) Extract minimum and maximum salary from salary_range ("20000-28000")
def get_min_salary(salary_range):
    if salary_range is None:
        return None
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", salary_range)
    return int(m.group(1)) if m else None


def get_max_salary(salary_range):
    if salary_range is None:
        return None
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", salary_range)
    return int(m.group(2)) if m else None


min_salary_udf = F.udf(get_min_salary, LongType())
max_salary_udf = F.udf(get_max_salary, LongType())

df = (df.withColumn("min_salary", min_salary_udf("salary_range"))
      .withColumn("max_salary", max_salary_udf("salary_range")))

df.filter(F.col("salary_range").isNotNull()) \
  .select("job_id", "salary_range", "min_salary", "max_salary", "job_type").show(10)

with_text = df.filter(F.col("salary_range").isNotNull()).count()
parsed = df.filter(F.col("min_salary").isNotNull()).count()
print("Ads with a salary_range value:", with_text)
print("Ads where min and max salary were extracted:", parsed)
print("Ads with a salary_range that could not be extracted:", with_text - parsed)
df.filter(F.col("salary_range").isNotNull() & F.col("min_salary").isNull()) \
  .select("salary_range").distinct().orderBy("salary_range").show(40, truncate=False)


def salary_stats(col):
    return (df.filter(F.col(col).isNotNull())
            .groupBy("job_type")
            .agg(F.count("*").alias("num_ads"),
                 F.round(F.mean(col), 2).cast("decimal(18,2)").alias("mean"),
                 F.round(F.median(col), 2).cast("decimal(18,2)").alias("median"),
                 F.round(F.stddev(col), 2).cast("decimal(18,2)").alias("std_dev"))
            .orderBy("job_type"))


print("Statistics of minimum salary:")
salary_stats("min_salary").show(truncate=False)
print("Statistics of maximum salary:")
salary_stats("max_salary").show(truncate=False)

print("Five largest maximum salaries:")
(df.filter(F.col("max_salary").isNotNull())
 .orderBy(F.desc("max_salary"), "job_id")
 .select("job_id", "salary_range", "max_salary", "job_type")
 .show(5, truncate=False))

# Q2(c) Salary distributions on the original and log scales
rows = (df.filter(F.col("min_salary").isNotNull())
        .select("job_type", "min_salary", "max_salary").collect())
data = {}
for jt in ["Fraudulent", "Genuine"]:
    data[jt] = {"min": np.array([r["min_salary"] for r in rows if r["job_type"] == jt], dtype=float),
                "max": np.array([r["max_salary"] for r in rows if r["job_type"] == jt], dtype=float)}
colors = {"Fraudulent": "tab:red", "Genuine": "tab:blue"}

for jt in data:
    print(jt, "- ads:", len(data[jt]["min"]),
          "| min salary = 0:", int((data[jt]["min"] == 0).sum()),
          "| max salary = 0:", int((data[jt]["max"] == 0).sum()))

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for col, key in enumerate(["min", "max"]):
    for row, scale in enumerate(["linear", "log"]):
        ax = axes[row][col]
        groups = [data[jt][key] if scale == "linear" else data[jt][key][data[jt][key] > 0]
                  for jt in ["Fraudulent", "Genuine"]]
        bp = ax.boxplot(groups, patch_artist=True)
        for patch, jt in zip(bp["boxes"], ["Fraudulent", "Genuine"]):
            patch.set_facecolor(colors[jt]); patch.set_alpha(0.5)
        ax.set_xticklabels(["Fraudulent", "Genuine"])
        ax.set_yscale(scale)
        ax.set_ylabel("Salary" + (" (log scale, values > 0)" if scale == "log" else ""))
        ax.set_title(("Minimum" if key == "min" else "Maximum") + " salary, " +
                     ("original scale" if scale == "linear" else "log scale"))
fig.tight_layout()
fig.savefig(os.path.join(base, "outputs", "q2c_boxplots.png"), dpi=130)

fig, axes = plt.subplots(4, 2, figsize=(11, 13))
for col, jt in enumerate(["Fraudulent", "Genuine"]):
    for r, (key, scale) in enumerate([("min", "linear"), ("max", "linear"),
                                      ("min", "log"), ("max", "log")]):
        ax = axes[r][col]
        values = data[jt][key]
        if scale == "linear":
            ax.hist(values, bins=50, color=colors[jt], alpha=0.6)
            ax.set_xlabel("Salary")
        else:
            values = values[values > 0]
            ax.hist(values, bins=np.logspace(0, 9.2, 40), color=colors[jt], alpha=0.6)
            ax.set_xscale("log")
            ax.set_xlabel("Salary (log scale, values > 0)")
        ax.set_ylabel("Number of ads")
        ax.set_title(f"{jt}: " + ("minimum" if key == "min" else "maximum") + " salary, " +
                     ("original scale" if scale == "linear" else "log scale"))
fig.tight_layout()
fig.savefig(os.path.join(base, "outputs", "q2c_histograms.png"), dpi=130)

summary = (df.filter(F.col("min_salary").isNotNull())
           .groupBy("job_type")
           .agg(F.count("*").alias("num_ads"),
                F.sum((F.col("min_salary") == 0).cast("int")).alias("min_is_zero"),
                F.sum((F.col("min_salary") < 1000).cast("int")).alias("min_below_1000"),
                F.sum((F.col("max_salary") > 1000000).cast("int")).alias("max_above_1m"))
           .withColumn("pct_above_1m", F.round(F.col("max_above_1m") / F.col("num_ads") * 100, 2))
           .orderBy("job_type"))
summary.show(truncate=False)

# Q2(d) Remove extreme salaries with the IQR method
iqr_stats = (df.filter(F.col("min_salary").isNotNull())
             .groupBy("job_type")
             .agg(F.percentile("min_salary", 0.25).alias("q1_min"),
                  F.percentile("min_salary", 0.75).alias("q3_min"),
                  F.percentile("max_salary", 0.25).alias("q1_max"),
                  F.percentile("max_salary", 0.75).alias("q3_max"))
             .withColumn("iqr_min", F.col("q3_min") - F.col("q1_min"))
             .withColumn("iqr_max", F.col("q3_max") - F.col("q1_max"))
             .orderBy("job_type"))
print("Quartiles and IQR of minimum salary:")
iqr_stats.select("job_type", "q1_min", "q3_min", "iqr_min").show(truncate=False)
print("Quartiles and IQR of maximum salary:")
iqr_stats.select("job_type", "q1_max", "q3_max", "iqr_max").show(truncate=False)

bounds = (iqr_stats
          .withColumn("lower_min", F.col("q1_min") - 1.5 * F.col("iqr_min"))
          .withColumn("upper_min", F.col("q3_min") + 1.5 * F.col("iqr_min"))
          .withColumn("lower_max", F.col("q1_max") - 1.5 * F.col("iqr_max"))
          .withColumn("upper_max", F.col("q3_max") + 1.5 * F.col("iqr_max"))
          .select("job_type", "lower_min", "upper_min", "lower_max", "upper_max"))
print("IQR bounds:")
bounds.show(truncate=False)

df_iqr = (df.filter(F.col("min_salary").isNotNull())
          .join(bounds, on="job_type")
          .filter((F.col("min_salary") >= F.col("lower_min")) & (F.col("min_salary") <= F.col("upper_min")) &
                  (F.col("max_salary") >= F.col("lower_max")) & (F.col("max_salary") <= F.col("upper_max"))))
before = df.filter(F.col("min_salary").isNotNull()).groupBy("job_type").count().withColumnRenamed("count", "before")
after = df_iqr.groupBy("job_type").count().withColumnRenamed("count", "after")
print("Number of postings before and after removing extreme values:")
(before.join(after, "job_type")
 .withColumn("removed", F.col("before") - F.col("after"))
 .withColumn("pct_removed", F.round(F.col("removed") / F.col("before") * 100, 2))
 .orderBy("job_type").show())

rows2 = df_iqr.select("job_type", "min_salary", "max_salary").collect()
clean = {}
for jt in ["Fraudulent", "Genuine"]:
    clean[jt] = {"min": np.array([r["min_salary"] for r in rows2 if r["job_type"] == jt], dtype=float),
                 "max": np.array([r["max_salary"] for r in rows2 if r["job_type"] == jt], dtype=float)}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for col, key in enumerate(["min", "max"]):
    bp = axes[col].boxplot([clean["Fraudulent"][key], clean["Genuine"][key]], patch_artist=True)
    for patch, jt in zip(bp["boxes"], ["Fraudulent", "Genuine"]):
        patch.set_facecolor(colors[jt]); patch.set_alpha(0.5)
    axes[col].set_xticklabels(["Fraudulent", "Genuine"])
    axes[col].set_ylabel("Salary")
    axes[col].set_title(("Minimum" if key == "min" else "Maximum") + " salary after IQR filtering")
fig.tight_layout()
fig.savefig(os.path.join(base, "outputs", "q2d_boxplots.png"), dpi=130)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for col, jt in enumerate(["Fraudulent", "Genuine"]):
    for r, key in enumerate(["min", "max"]):
        axes[r][col].hist(clean[jt][key], bins=30, color=colors[jt], alpha=0.6)
        axes[r][col].set_xlabel("Salary")
        axes[r][col].set_ylabel("Number of ads")
        axes[r][col].set_title(f"{jt}: " + ("minimum" if key == "min" else "maximum") +
                               " salary after IQR filtering")
fig.tight_layout()
fig.savefig(os.path.join(base, "outputs", "q2d_histograms.png"), dpi=130)


# Q2(e) Reusable outlier functions with sensible custom salary limits
def calculate_bounds(data, column, multiplier=1.5):
    q1, q3 = data.select(F.percentile(column, 0.25), F.percentile(column, 0.75)).first()
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def flag_outliers(data, column, bound_lower, bound_upper, custom_minimum, custom_maximum):
    lower_limit = min(bound_lower, custom_minimum)
    upper_limit = max(bound_upper, custom_maximum)
    return data.withColumn(column + "_outlier",
                           (F.col(column) < lower_limit) | (F.col(column) > upper_limit))


CUSTOM_MIN = 12000
CUSTOM_MAX = 260000
valid = df.filter(F.col("min_salary").isNotNull())
flagged_parts = []
print("Limits used for flagging outliers:")
for jt in ["Fraudulent", "Genuine"]:
    part = valid.filter(F.col("job_type") == jt)
    for col in ["min_salary", "max_salary"]:
        lo, hi = calculate_bounds(part, col)
        print(f"{jt:<11}{col:<11} IQR bounds: ({lo:>10.1f}, {hi:>10.1f})   "
              f"limits used: ({min(lo, CUSTOM_MIN):>10.1f}, {max(hi, CUSTOM_MAX):>10.1f})")
        part = flag_outliers(part, col, lo, hi, CUSTOM_MIN, CUSTOM_MAX)
    part = part.withColumn("is_outlier", F.col("min_salary_outlier") | F.col("max_salary_outlier"))
    flagged_parts.append(part)

flagged = flagged_parts[0].unionByName(flagged_parts[1])
print("Number of outliers flagged and removed:")
(flagged.groupBy("job_type")
 .agg(F.count("*").alias("before"),
      F.sum(F.col("is_outlier").cast("int")).alias("removed"))
 .withColumn("after", F.col("before") - F.col("removed"))
 .orderBy("job_type").show())
print("Postings that were flagged as outliers (first 20):")
(flagged.filter("is_outlier")
 .select("job_id", "salary_range", "job_type")
 .orderBy("job_type", "job_id").show(20, truncate=False))

df_final = flagged.filter(~F.col("is_outlier"))


def stats_table(data, col):
    return (data.groupBy("job_type")
            .agg(F.count("*").alias("num_ads"),
                 F.round(F.mean(col), 2).cast("decimal(18,2)").alias("mean"),
                 F.round(F.median(col), 2).cast("decimal(18,2)").alias("median"),
                 F.round(F.stddev(col), 2).cast("decimal(18,2)").alias("std_dev"))
            .orderBy("job_type"))


print("Statistics of minimum salary after removing outliers:")
stats_table(df_final, "min_salary").show(truncate=False)
print("Statistics of maximum salary after removing outliers:")
stats_table(df_final, "max_salary").show(truncate=False)


# Q2(f) Most frequent bi-grams and tri-grams in job descriptions
def to_lower_case(column):
    return F.lower(column)


def remove_non_letters(column):
    return F.regexp_replace(column, r"[^\p{L}]+", " ")


def remove_extra_spaces(column):
    return F.trim(F.regexp_replace(column, r"\s+", " "))


def clean_text(column):
    return remove_extra_spaces(remove_non_letters(to_lower_case(column)))


text_df = (df.withColumn("clean_description", clean_text(F.coalesce(F.col("description"), F.lit(""))))
           .select("job_type", "description", "clean_description"))
text_df.select("job_type", "clean_description").show(3, truncate=90)

tokenizer = RegexTokenizer(inputCol="clean_description", outputCol="words", pattern=r"\s+")
words_df = tokenizer.transform(text_df)


def top_ngrams(data, n, job_type, top=10):
    ngram = NGram(n=n, inputCol="words", outputCol="ngrams")
    return (ngram.transform(data.filter(F.col("job_type") == job_type))
            .select(F.explode("ngrams").alias("ngram"))
            .groupBy("ngram").count()
            .orderBy(F.desc("count"), "ngram")
            .limit(top)
            .collect())


for n, name in [(2, "bi-grams"), (3, "tri-grams")]:
    for jt in ["Fraudulent", "Genuine"]:
        print(f"Top 10 {name} for {jt} job postings:")
        for row in top_ngrams(words_df, n, jt):
            print((row["ngram"], row["count"]))

spark.stop()
