# Prediction of Commercial Insurance Payments for Surgical Procedure Using DataRobot
Project Objective: Our goal is to improve on the previous work that has been done by Johnson and Johnson in terms of predicting commerical insurance payments for surgical procedures. J&J used DataRobot to approach this problem initially, and the result was a Gradient Boosting on ElasticNet model. Our objective is to create a model that has equivalent or better accuracy than their original model while also providing better flexibility and visibility. Using DataRobot prevented the original model from being very flexible, so we are looking to create additional flexibility in ways like setting limitations on the target variable.


####  Authors: + Ryan Rogers (rjr2168)(Team Captain) + Shruti Kaushal (sk4963) + Parv Joshi (pj2384) + Sarthak Arora (sa4001) + Tyler Marshall (tam2197) 
####  Sponsor/Mentor: - Cindy Tong from Johnson & Johnson 
####  CA: - Cathy Li 
####  Instructor: - Adam Kelleher

#### Prospective Data Sources:
* [S0101](https://data.census.gov/cedsci/table?t=Age%20and%20Sex&g=0100000US%243100000&y=2021&tid=ACSST1Y2021.S0101) - Demographic Profile information for MSAs
* [S1811](https://data.census.gov/cedsci/table?q=S1811&g=0100000US%243100000&tid=ACSST1Y2021.S1811) - Economic characteristics for civilian population, broken down by MSA
* [S0201](https://data.census.gov/cedsci/table?q=foreign&g=0100000US%243100000&tid=ACSSPP1Y2021.S0201) - Selected population profile (demographic & immigration/citizenship data), broken down by MSA
* [SAHIE](https://www.census.gov/data/datasets/time-series/demo/sahie/estimates-acs.html) - Small Area Health Insurance Estimates Census Data by State/County
* [SAPIE](https://www.census.gov/data/datasets/2020/demo/saipe/2020-state-and-county.html) - Poverty and Income Estimates by State/County
* [Business MSA](https://www.census.gov/data/datasets/2019/econ/cbp/2019-cbp.html) - Business pattern data at MSA level
