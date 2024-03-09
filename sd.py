
# Use for test purpus
import  pickle
import pandas as pd
import datetime
d=datetime.datetime.strptime('2018-10-1','%Y-%m-%d')

# model=pickle.load(open("LinerRegration_model.pkl","rb"))
model1=pickle.load(open("RandomForestRegressor_model.pkl","rb"))
t=pickle.load(open("RandomForestRegressor_transfrom.pkl","rb"))
t_d=t.transform( pd.DataFrame([['Skoda Superb LK 1.8 TSI',d,'Comprehensive','Petrol',5,30615,'First Owner','Automatic',17.40,999.0,8314.0]],columns=['car_name','registration_year','insurance_validity','fuel_type','seats','kms_driven','ownsership','transmission','mileage(kmpl)','engine(cc)','torque(Nm)']))
print(model1.predict(t_d))
