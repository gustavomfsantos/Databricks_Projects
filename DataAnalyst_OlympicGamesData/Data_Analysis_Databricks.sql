-- Databricks notebook source
-- MAGIC %md
-- MAGIC This Notebook is designated build a Data Analysis project using Olympic Games Athletes. This data starts at 1896 and goes until the 2016 games. 
-- MAGIC
-- MAGIC The project starts at Loading the data files stored in Dropox, transforming them into Pandas Dataframes and later on Relational Tables. The origin form of the data is denormalized and, as good exercise and best practice regarding data engineer, we will normalized in order to attend future needs to create a Data Warehouse with it. But queries will be done on the denormalized table. Most commands will use python syntax but for queries the priority will be use SQL language.
-- MAGIC
-- MAGIC The first step is to understand our data, what does it contain, what kind of information and how reliable is it. Then, establish some questions that the data might be able to answer, make assumptions and create hypothesis and check if they are correct. We will do all that and use different approaches depending on the problem/question that we are looking at. 
-- MAGIC
-- MAGIC The final goal is be able to discover insights from the data and inform those in a direct and understandable way to an audience that is not used to Data or Statistical terms.
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Our first step is load the dataset and take a brief look at the data.
-- MAGIC
-- MAGIC During the code execution and the whole project, more explanation will be provided, and most code sections will have a comment to explain what is happening.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC print('Importing Libraries')
-- MAGIC import pandas as pd
-- MAGIC import requests
-- MAGIC from io import BytesIO
-- MAGIC from zipfile import ZipFile
-- MAGIC
-- MAGIC from pyspark.sql import SparkSession
-- MAGIC from pyspark.sql import DataFrame
-- MAGIC
-- MAGIC import matplotlib.pyplot as plt
-- MAGIC import seaborn as sns
-- MAGIC #import missingno as msno
-- MAGIC from sklearn.preprocessing import StandardScaler
-- MAGIC
-- MAGIC print('Libraries Imported')

-- COMMAND ----------

-- MAGIC %python
-- MAGIC print('There are two files, a zip file that we will get and unzip to create a pandas Dataframe with the unziped csv file')
-- MAGIC print('The other file is a csv file that we will load as pandas DataFrame as well. Both files are in dropbox and we will use the link to reach them')
-- MAGIC
-- MAGIC zip_url = 'https://dl.dropboxusercontent.com/scl/fi/9i9dmwkyikwyu2gvwvf5h/athlete_events.csv.zip?rlkey=lb9sglah6vltvw2aoj79rhyz9&dl=0'
-- MAGIC
-- MAGIC # Download the zip file
-- MAGIC response = requests.get(zip_url)
-- MAGIC
-- MAGIC # Check if the request was successful (status code 200)
-- MAGIC if response.status_code == 200:
-- MAGIC     # Extract the zip file contents into a BytesIO object
-- MAGIC     with ZipFile(BytesIO(response.content)) as zip_file:
-- MAGIC         csv_file_name = zip_file.namelist()[0]
-- MAGIC
-- MAGIC         # Read the CSV file into a Pandas DataFrame
-- MAGIC         with zip_file.open(csv_file_name) as csv_file:
-- MAGIC             df_pandas1 = pd.read_csv(csv_file)
-- MAGIC else:
-- MAGIC     print(f"Failed to download the zip file. Status code: {response.status_code}")
-- MAGIC
-- MAGIC
-- MAGIC csv_url = 'https://dl.dropboxusercontent.com/scl/fi/n2an3opvzbuhfz1donb04/noc_regions.csv?rlkey=2kq70zpjw2wakdkf4tqzp2dfl&dl=0'
-- MAGIC
-- MAGIC # Read the CSV file into a Pandas DataFrame
-- MAGIC df_pandas2 = pd.read_csv(csv_url)
-- MAGIC
-- MAGIC print('Files Loaded!')

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Files Loaded and Pandas Dataframes created Successfully!
-- MAGIC
-- MAGIC Let's take a look how the data is distriuited, what those columns contain, seek out for missing values and other preliminary analysis. All of that using the Pandas dataframe and Python language

-- COMMAND ----------

-- MAGIC %python
-- MAGIC print('Display dataframes and see data types of each column')
-- MAGIC display(df_pandas1)
-- MAGIC display(df_pandas2)
-- MAGIC
-- MAGIC print(df_pandas1.info())
-- MAGIC print(df_pandas2.info())

-- COMMAND ----------

-- MAGIC %md
-- MAGIC We can see that both dataframes have some null values. Before digging into the columns aspects and normalization of the data, let's take deep look into those Not Available Numbers, plotting the Null values in both dataframes
-- MAGIC

-- COMMAND ----------

-- MAGIC %python
-- MAGIC for index,df in enumerate([df_pandas1, df_pandas2]):
-- MAGIC     print(f'Plotting graph for df_pandas{index + 1}')
-- MAGIC     null_counts = df.isnull().sum()
-- MAGIC
-- MAGIC     plt.figure(figsize=(10, 6))
-- MAGIC     bars = plt.bar(null_counts.index, null_counts.values, color='skyblue')
-- MAGIC
-- MAGIC     # Add labels on top of each bar
-- MAGIC     for bar in bars:
-- MAGIC         yval = bar.get_height()
-- MAGIC         plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom')
-- MAGIC
-- MAGIC     plt.title(f'Total Null Values per Column')
-- MAGIC     plt.xlabel('Columns')
-- MAGIC     plt.ylabel('Null Count')
-- MAGIC     plt.show()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC There are some assumptions that we can make that are related to those null values.
-- MAGIC
-- MAGIC Regarding the first dataframe, the column Medal has probably a lot of null because only the first three athletes in each competition get a medal, so most athletes won't get any medals.
-- MAGIC
-- MAGIC Height and weight data was proabably hard to collect or keep in older editions. Same for age, but in a minor degree. Information about teams, sports, year, city, and sex are complete
-- MAGIC
-- MAGIC In the second dataframe, only three regions have null values. Before move foward, let's found out those Null values. Notes are mostly not usable information, so there is no cost to having a lot of null values.
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- MAGIC %python
-- MAGIC print('Now lets see which are the 3 null values in dataframe2')
-- MAGIC nan_rows = df_pandas2[df_pandas2['region'].isna()][['NOC','notes']]
-- MAGIC print(nan_rows)
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC The teams that don't have regions are ROT, which has a note to inform that it is the Refugee Olympic Team; TUV, which has a note saying 'Tuvalu', a country in Oceania; and UNK, which has no notes, but after some google, we found out that it is for Kosovo, a country with limited recognition. Lets limit our use to only NOC.
-- MAGIC
-- MAGIC Now a more complex analysis on the dataframe1. Using python packages, let's explore the correlation heatmap to see if we can find any clear correlation between our columns and plot boxplot to see how the columns are distributed, checking for outliers.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC print('Lets take a look at the dataframe1')
-- MAGIC for index,df in enumerate([df_pandas1]):
-- MAGIC     print(f'Plotting graph for df_pandas{index + 1}')
-- MAGIC     corr_matrix = df.corr()
-- MAGIC     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5)
-- MAGIC     plt.show()
-- MAGIC     for column in df.select_dtypes(include=['int64', 'float64']).columns:
-- MAGIC         sns.boxplot(x=column, data=df)
-- MAGIC         plt.show()
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC So we can see a clear correlation between height and weight, which is expected. Regarding the outliers, it is an expected boxplot. Most athletes have a similar body and close age, but in a heterogenous event such as the Olympic Games, with a lot of sports, we will have athletes of various ages, heights, and weights. The boxplot for feature year is looking trickier. Let's take a closer look.
-- MAGIC

-- COMMAND ----------

-- MAGIC %python
-- MAGIC max_value_year = df_pandas1['Year'].max()
-- MAGIC min_value_year = df_pandas1['Year'].min()
-- MAGIC
-- MAGIC print(f"Max value in 'Year': {max_value_year}")
-- MAGIC print(f"Min value in 'Year': {min_value_year}")

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Everything looks good.
-- MAGIC
-- MAGIC Let's go to the next steps now that we know what our dataset looks like. Let's create a SQL table using Spark. That way, we can make queries using SQL commands. But we will keep the Pandas dataframe for further modeling. Our table will be a join from our two Pandas dataframes. The code below does all the work, explaining each step.
-- MAGIC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC With all this primary analysis done, the next step is create a database and tables in order to permit SQL queries on the dataset. During this task, lets keep the denormalized table for queries. But we will keep the Pandas dataframe for further modeling. 
-- MAGIC
-- MAGIC Our first table will be a join from our two Pandas dataframes using Spark library. The code below does all the work, explaining each step.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC
-- MAGIC # Create a Spark session
-- MAGIC spark = SparkSession.builder.appName("example").getOrCreate()
-- MAGIC spark.conf.set("spark.sql.legacy.allowNonEmptyLocationInCTAS", "true")
-- MAGIC # Check if 'df' is a Spark DataFrame
-- MAGIC if isinstance(df_pandas1, DataFrame):
-- MAGIC     print("df is a Spark DataFrame")
-- MAGIC else:
-- MAGIC     print("df is not a Spark DataFrame")
-- MAGIC     df_spark1 = spark.createDataFrame(df_pandas1)
-- MAGIC
-- MAGIC if isinstance(df_pandas2, DataFrame):
-- MAGIC     print("df is a Spark DataFrame")
-- MAGIC else:
-- MAGIC     print("df is not a Spark DataFrame")
-- MAGIC     df_spark2 = spark.createDataFrame(df_pandas2)
-- MAGIC
-- MAGIC #Create table using spark DF
-- MAGIC df_spark1.createOrReplaceTempView("table1")
-- MAGIC df_spark2.createOrReplaceTempView("table2")
-- MAGIC
-- MAGIC # Define the SQL query with the JOIN clause
-- MAGIC sql_query = """
-- MAGIC SELECT 
-- MAGIC     ID,
-- MAGIC     Name,
-- MAGIC     Sex,
-- MAGIC     Age,
-- MAGIC     Height,
-- MAGIC     Weight,
-- MAGIC     Team,
-- MAGIC     region,
-- MAGIC     table1.NOC,
-- MAGIC     Games,
-- MAGIC     Year,
-- MAGIC     Season,
-- MAGIC     City,
-- MAGIC     Sport,
-- MAGIC     Event,
-- MAGIC     Medal,
-- MAGIC     notes
-- MAGIC FROM 
-- MAGIC     table1
-- MAGIC LEFT JOIN 
-- MAGIC     table2 
-- MAGIC ON 
-- MAGIC     table1.NOC = table2.NOC
-- MAGIC """
-- MAGIC
-- MAGIC # Execute the SQL query
-- MAGIC denormalized_table = spark.sql(sql_query)
-- MAGIC
-- MAGIC #Create a table based on the query output
-- MAGIC denormalized_table.createOrReplaceTempView("denormalized_table")
-- MAGIC
-- MAGIC print('Now we have a Table named denormalized_table')
-- MAGIC
-- MAGIC
-- MAGIC

-- COMMAND ----------

-- Creating a table in databricks with the result from the merge above!
DROP TABLE IF EXISTS spark_catalog.default.denormalized;
CREATE TABLE denormalized
USING parquet
OPTIONS (
  'path' '/mnt/dbfs',
  'overwrite' 'true'  -- Use 'overwrite' to replace the table if it already exists
)
AS
SELECT *
FROM denormalized_table

-- COMMAND ----------

-- Check the tables in our enviroment
SHOW TABLES IN spark_catalog.default;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Now we have a Delta Table with our dataset. We can use this table to query more specific analysis and get more insights. Our first impressions gave us some idea about the dataset. We are able to make some questions and assumptions related to the data.
-- MAGIC
-- MAGIC Before dive into a possible correlations, lets take a deeper look at previous discovers that we made, for example:
-- MAGIC - Is the refugge team made up of many nationalities? Does it have a main nation on this team? How many medals do they have, and in how many editions did they participate?
-- MAGIC - How football teams were organized: female and male teams. When they started playing in the Olympic games. Once they were introduced, were they always there? How many players can a nation have, and is it always like that?
-- MAGIC
-- MAGIC Then we go further in a analysis that can bring market value. Since we have winner information at each event and physical attributes from athletes, we can look for any kind of correlation between those features. Important to note that each sport has specific ideal body type, so will be wise to make this analysis grouping sports in order to not compare different sports. If we manage to find any significant correlation, we can build a predictive model to forecast possible winner in each sports. In modern times, sports betting has become a huge market with the potential to become even bigger. A model that can help predict a possible winner can reduce the uncertain risk of sporting bets and maximize profit (questionable if betting is profitable).
-- MAGIC
-- MAGIC Another approach we can use is a wider one. Instead of looking at the athletes level we can look at countries level. We can aggregate all atheltes with same nationality and make contry analysis. We know that in the history of Olympic games there is a dominance for a few countries. But a nice hypothesis to test is, assuming that the global population distribution is normal with constant variance, countries that have a bigger population will have more talented citizen, assuming that an athlete is born with talents that average people don't have. Assuming that and ignoring others influences such as trainning time, infrastructure and financial support, countries with larger population will perform better than countries will small population because they will have, in absolute number, more talented individuals.
-- MAGIC

-- COMMAND ----------

SELECT 
  *
FROM 
  denormalized
WHERE 
  notes = 'Refugee Olympic Team'

-- COMMAND ----------

-- MAGIC %md
-- MAGIC We are not able to get the nationalities from the information that we have. But we can say that they were only assembled once, for the 2016 games, and got no medals.
-- MAGIC
-- MAGIC Let's take a look at the sex distribution and sports distribution.

-- COMMAND ----------

SELECT 
  Sex,
  Sport,
  COUNT(ID)
FROM 
  denormalized
WHERE 
  notes = 'Refugee Olympic Team'
GROUP BY 
  Sex,
  Sport

-- COMMAND ----------

-- MAGIC %md
-- MAGIC So, we had only 3 sports for the refugee team: athletics, judo,  and swimming. In Judo and swimming, the male and female competitors were equal, but in athletics, there were more male athletes than female athletes. In total, the refugee team had 12 athletes: 7 males and 5 females.
-- MAGIC
-- MAGIC Now let's take a look at the football data. THe curiosity around football data is related to my nationality, Brazilian.
-- MAGIC

-- COMMAND ----------

-- Lets get the max value of each edition/year. Also some other metrics to see how those number behaved each olimpic games.
SELECT 
  MAX(players),
  MIN(players),
  ROUND(AVG(players),0),
  MODE(players), 
  COUNT(NOC),
  Year,
  Sex
FROM
  (
  SELECT 
    NOC,
    Sex,
    Year,
    COUNT(ID) AS players
  FROM 
    denormalized
  WHERE
    Sport = 'Football'
  GROUP BY 
    NOC,
    Sex,
    Year

  ORDER BY
    players DESC
  ) AS players_count
GROUP BY
  Year,
  Sex
ORDER BY
  Year ASC

-- COMMAND ----------

-- MAGIC %md
-- MAGIC With the query above, we can assume that only the 1932 edition didn't have a football competition. Was that a year without Olympic games or just lacking football? We will check later. Also, the female team was introduced in 1996.
-- MAGIC
-- MAGIC First editions allowed more players in teams, but recent editions seem to have an 18-player limit. Also, the competitiors limit for males is 16 countries and 12 for females, taking into account the last editions.
-- MAGIC
-- MAGIC Now let's see if there was a 1932 edition.
-- MAGIC

-- COMMAND ----------

SELECT 
  Sport,
  COUNT(DISTINCT(NOC))
FROM 
  denormalized
WHERE
  Year = 1932
GROUP BY
  Sport

-- COMMAND ----------

-- MAGIC %md
-- MAGIC There was a 1932 edition. Let's see the right before and right after editions to compare the number of sports and countries participating.

-- COMMAND ----------

SELECT 
  Sport,
  Year,
  COUNT(DISTINCT(NOC))
FROM 
  denormalized
WHERE
  Year = 1928 OR Year = 1932 OR Year = 1936
GROUP BY
  Sport,
  Year
ORDER BY 
  Sport, 
  Year

-- COMMAND ----------

-- MAGIC %md
-- MAGIC So the 1932 edition had the same sports as the last edition, 1928, but with fewer nations accounting across the sports. And the next edition, 1936, had much more athletes representing nations and more sports.
-- MAGIC
-- MAGIC We can say that the 1932 edition was not as complete as usual. As final analysis using this football data, let's see who has the most medals in this competition. Male and Female categories and aggregate
-- MAGIC

-- COMMAND ----------

-- Lets get all nation that got at least one medal. First lets do for Male and then Female. Since the visualization tool does not allow filter by column, 2 commands are necessary
SELECT 
  NOC,
  Sex,

  SUM(CASE WHEN Medal = 'Gold' THEN 1 ELSE 0 END) AS gold_medal_total,
  SUM(CASE WHEN Medal = 'Silver' THEN 1 ELSE 0 END) AS silver_medal_total,
  SUM(CASE WHEN Medal = 'Bronze' THEN 1 ELSE 0 END) AS bronze_medal_total,
  (bronze_medal_total + silver_medal_total + gold_medal_total) AS total_medals
FROM
(
SELECT 
  NOC,
  Sex,
  Year,
  Medal
FROM 
  denormalized
WHERE
  Sport = 'Football' AND
  Sex = 'M'
GROUP BY 
  NOC,
  Sex,
  Year,
  Medal
HAVING 
  Medal IS NOT NULL
) AS medals_football
GROUP BY
  NOC,
  Sex

-- COMMAND ----------

SELECT 
  NOC,
  Sex,

  SUM(CASE WHEN Medal = 'Gold' THEN 1 ELSE 0 END) AS gold_medal_total,
  SUM(CASE WHEN Medal = 'Silver' THEN 1 ELSE 0 END) AS silver_medal_total,
  SUM(CASE WHEN Medal = 'Bronze' THEN 1 ELSE 0 END) AS bronze_medal_total,
  (bronze_medal_total + silver_medal_total + gold_medal_total) AS total_medals
FROM
(
SELECT 
  NOC,
  Sex,
  Year,
  Medal
FROM 
  denormalized
WHERE
  Sport = 'Football' AND
  Sex = 'F'
GROUP BY 
  NOC,
  Sex,
  Year,
  Medal
HAVING 
  Medal IS NOT NULL
) AS medals_football
GROUP BY
  NOC,
  Sex

-- COMMAND ----------

-- Now a graph will both sex to plot total medals
SELECT 
  NOC,
  Sex,

  SUM(CASE WHEN Medal = 'Gold' THEN 1 ELSE 0 END) AS gold_medal_total,
  SUM(CASE WHEN Medal = 'Silver' THEN 1 ELSE 0 END) AS silver_medal_total,
  SUM(CASE WHEN Medal = 'Bronze' THEN 1 ELSE 0 END) AS bronze_medal_total,
  (bronze_medal_total + silver_medal_total + gold_medal_total) AS total_medals
FROM
(
SELECT 
  NOC,
  Sex,
  Year,
  Medal
FROM 
  denormalized
WHERE
  Sport = 'Football' 
GROUP BY 
  NOC,
  Sex,
  Year,
  Medal
HAVING 
  Medal IS NOT NULL
) AS medals_football
GROUP BY
  NOC,
  Sex

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Naturally, as female competiton was introduced way later, there is way more medals for Male Teams. And that the same reason why there is fewer contries on the medal distribution. Hopefully, the next editions will bring more variety to medalist rol. In the mens distribution, we can see some countries that no longer exists, such as Yugoslavia that was separated into several countries in 1992 and GDR that was the East Germany representation during cold war period. Another observation is how Brazilian team has only one gold medal (won in 2016 edition) even though is considered the major country in the sport and has 5 World Cups. The fact the Olympic games is limited to 23 year age athletes might influence on this, assuming that Brazilian support to young talents is below the average support on other countries and Brazilian football athletes development occurs when they migrate do play on European cenario. Those are only hypothesis that we might work later but this dataset is not enough to check those hypothesis.
-- MAGIC
-- MAGIC With that, we finish our sports analysis. Let's take a look at the athlete's data, focusing on weight and height.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Dealing with athletes, we can assume that some athletes might compete in more than one Olympic Games edition and, even in same edition, can compete in more than one event. So, in order to analyse athlete individually, we need to consider only physical attributes and personal info, dropping duplicates and counting the total medals for each athlete and keep the sport info, because event info will generate duplicates. The approach here is to get the physical attributes, sports and medals and use GROUP BY clause to group by Sport and Medals, which will get groups for each sport and inside each sport, a separation between gold, silver, bronze and not medals (NULL values). And with the group by we will get the average Height and Weight from each group. An important step is to drop all athletes that dont have physical attributes available. They are not representative and might contaminate our analysis.
-- MAGIC
-- MAGIC The following SQL commands starts our analysis.

-- COMMAND ----------

DROP TABLE IF EXISTS spark_catalog.default.attributes_medals;

CREATE TABLE  attributes_medals
USING  PARQUET
OPTIONS (
  'path' '/mnt/dbfs1',
  'overwrite' 'true'  -- Use 'overwrite' to replace the table if it already exists
)
AS
SELECT *
FROM
  (
  SELECT 
    AVG(Height) AS average_height,
    AVG(Weight) AS average_weight,
    AVG(Weight)/AVG(Height) AS Weight_Porpotion_by_Height,
    Sport,
    Sex,
    Medal 
  FROM  
    (
    SELECT 
      ID, Height, Weight, Sex, Sport, Medal
    FROM 
      denormalized
    WHERE
      Height IS NOT NULL
      AND
      Weight IS NOT NULL
    ) AS medal_athletes
  GROUP BY
    Medal,
    Sport,
    Sex
  );


-- COMMAND ----------

SHOW TABLES IN spark_catalog.default;

-- COMMAND ----------

SELECT * FROM denormalized;

-- COMMAND ----------

SELECT *
FROM attributes_medals

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Now we have the table that we need to do our analysis regarding the attributes inside each sport and each category. Lets get some specific sports that are common sense where physical attributes might have some influence, such as volleyball, basketball, swimming and rowing. We expect that the medalists, specially the gold winners, have more height or weight in average than the Null group, that had no medals. First lets do for Male category then Female.
-- MAGIC

-- COMMAND ----------

SELECT 
  *
FROM 
  attributes_medals
WHERE
  (Sport = 'Basketball' OR Sport = 'Volleyball' OR Sport = 'Swimming' OR Sport = 'Rowing' )
  AND Sex = 'M'

-- COMMAND ----------

SELECT 
  *
FROM 
  attributes_medals
WHERE
  (Sport = 'Basketball' OR Sport = 'Volleyball' OR Sport = 'Swimming' OR Sport = 'Rowing' )
  AND Sex = 'F'

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Ok, it was according with our hypothesis but that is kind of common sense. Let's try to think about some sports where being small and lighter might be advantage. I can only come with Diving and Table Tennis. Since it is only two sport, let make the query with both sexs.
-- MAGIC

-- COMMAND ----------

SELECT 
  CONCAT(Sport, '_', Sex) AS Sport_Sex,
  average_height,
  average_weight,
  Weight_Porpotion_by_Height,
  Medal
FROM 
  attributes_medals
WHERE
  (Sport = 'Table Tennis' OR Sport = 'Diving')
 

-- COMMAND ----------

-- MAGIC %md
-- MAGIC For diving we have a clear pattern regarding weight, where lighter athletes perform better than heavier divers. But there is no evidence regarding height in diving sport. About Table Tenis, there is no clear dominance in height or weight regarding performance. It might have more sports where lighter and smaller athletes might dominate but that is not the full scope of this project.
-- MAGIC
-- MAGIC Assuming that Teams/Nations gather the best sportsman and sportswoman they have available, athletes or teams that have physical attributes that better adjust to sport the athlete is competing tend to perform better, on average. With that we finish the athlete/sport analysis. Let get into our final scope, get information at macro level and test our hypothesis in which populous countries will have better performance than those that are not populous. 
-- MAGIC
-- MAGIC A observation point is that when building Olyumpic Ranking during the Games, the first critery is Gold Medals. So for this analysis we will consider only Gold Medals as a performance metric. Another approach that will be used is to consider only the last 5 editions and sum the total for those 5 editions
-- MAGIC

-- COMMAND ----------



SELECT 
  NOC, SUM(medal_binary) as Medal_count
FROM
  (
  SELECT 
    ID, Medal, NOC, Year,
    CASE WHEN Medal = 'Gold' THEN 1 ELSE 0 END AS medal_binary
  FROM 
    denormalized
  WHERE Year BETWEEN 2000 AND 2016
  ) AS noc_medals
GROUP BY
  NOC
ORDER BY Medal_count DESC
LIMIT 50

-- COMMAND ----------

-- MAGIC %md
-- MAGIC The most dominant country is USA with almost twice gold medals as the Second place, Russia. Then we have Germany, China and Canada completing the top 5. According to https://www.worldometers.info/world-population/population-by-country/ 3 of those countries are among the top10 most populous in the world. And if we consider top30, 7 of the top10 countries in this chart are among them. 
-- MAGIC
-- MAGIC However, lets look at the opposite perpective. There are countries ranked among the last populous in the world that performed well in the last 5 Olympic games editions aggregate? Lets assume a threshold of 20 millions. That is close to the 60th most populous contry. So all countries that have less than 20 millions is considered not populous. In this chart we have Netherlands, Norway, Hungary, Sweden, Cuba, Denmark and Romania among the top20 performers on that 5 editions cutoff. So 7 countries, where some of then have specific weather (snow) that can benefit at some sports, but thats another analysis. I would say that eventhough those 7 countries, there is strong evidence that populous countries will perform better. Of course, that is a very simple correlation and, if we include other variables related to sport financial support and others, it might desapear or be weaker than we think.
