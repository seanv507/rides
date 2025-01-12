#%%
import pandas as pd 

Merily Kesküll
	
Aug 5, 2024, 5:34 PM (2 days ago)
	
to me

Hi again Sean,

 

Thank you again for your interest in Bolt and in our (Senior) Data Scientist position.

 

As I promised, I am sending you our home task. 

 

The goal of the task is to understand your strengths in the data science process. The home task consists of a problem that is similar to our day-to-day work. We will share a synthetic dataset with you and a task description, and we expect you to submit a rendered notebook showing your work. 

 

We will evaluate:

    The quality of your exploratory data analysis
    The coherence and structure of your ideas and code
    Your modeling approach, training, and validation methodology
    The applicability of your modeling technique to the real-life task you are trying to solve
    Your take on how the impact of the model should be evaluated in real-world interactions with our user base 

Please find the introduction to the task here:

 

Efficient supply allocation

The success of Bolt as a ride-hailing platform depends on a marketplace or efficient matching supply and demand in real-time. There are two sides in a ride-hailing marketplace: riders (demand) and drivers (supply). One of the challenges that we aim to solve is efficient supply allocation, so riders can always get a ride and drivers have stable earnings. Knowledge about how demand changes over time and space is crucial to comprehend supply dynamics. 

 

For this test task we expect you to:

 

    Explore the data and suggest a solution to guide the drivers towards areas with higher expected demand at given time and location
    Build and document a baseline model for your solution
    Describe how you would design and deploy such a model
    Describe how to communicate model recommendations to drivers
    Think through and describe the design of the experiment  that would validate your solution for live operations taking into account marketplace specifics

The goal of the task is to understand your strengths in the data science fundamentals and product thinking.
Data for the task:

The source data is approximately 630000 rows of synthetic ride demand data which resembles the real-life situation in the city of Tallinn:

    start_time - time when the order was made
    start_lat - latitude of the order's pick-up point
    start_lng - longitude of the order's pick-up point
    end_lat - latitude of the order's destination point
    end_lng - longitude of the order's destination point
    ride_value - how much monetary value is in this particular ride

# To do

1. null deviance
1. linear model
1. show tree/ residuals /  - sense checking xgboost... what sort of model is it building?
1. why doesn't using previous 15 minutes work?