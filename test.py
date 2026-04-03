import pandas as pd

data = pd.DataFrame({
    "Name":["porsche","bmw",'audi','toyota'],
    "speed":[310, 280, 290, 250]
})

max_speed = data['speed'].max()
car_name = data.loc[data["speed"] == max_speed, 'Name']
print("max speed = ", max_speed)
print("car name with max speed=", car_name[0])