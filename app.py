
from flask import Flask ,render_template,request
import pandas as pd
import datetime
import  pickle

model=pickle.load(open("RandomForestRegressor_model.pkl","rb"))
t=pickle.load(open("RandomForestRegressor_transfrom.pkl","rb"))

app=Flask(__name__)

car=pd.read_csv("clean.csv")

@app.route('/')
def index():
    car_name=car["car_name"].unique()
    
    register_year=sorted(car["registration_year"].unique())
    insurnce=car["insurance_validity"].unique()
    fule_type=car["fuel_type"].unique()
    seats=sorted(car["seats"].unique())
    kms_driven=car["kms_driven"].unique()
    ownsership=car["ownsership"].unique()
    transmission=car["transmission"].unique()
    mileage=car["mileage(kmpl)"].unique()
    engine=car['engine(cc)'].unique()
    torque=car["torque(Nm)"].unique()
    
    return render_template('index.html',name=car_name,year=register_year,insurnce=insurnce,fule=fule_type,seat=seats,kms_driven=kms_driven,ownsership=ownsership,transmission=transmission,mileage=mileage,engine=engine,torque=torque)

@app.route('/predict',methods=['POST'])
def predict():
    name=request.form.get("name")
    year=request.form.get("year")
    y=datetime.datetime.strptime(year,'%Y-%m-%d')
    insurnce=request.form.get("insurnce")
    fule=request.form.get("fule")
    seat=int(request.form.get("seat"))
    kms_driven=int(request.form.get("kms_driven"))
    ownsership=request.form.get("ownsership")
    transmission=request.form.get("transmission")
    mileage=int(request.form.get("mileage"))
    engine=int(request.form.get("engine"))
    torque=int(request.form.get("torque"))
    t_d=t.trnasform(pd.DataFrame([[name,y,insurnce,fule,seat,kms_driven,ownsership,transmission,mileage,engine,torque]],columns=['car_name','registration_year','insurance_validity','fuel_type','seats','kms_driven','ownsership','transmission','mileage(kmpl)','engine(cc)','torque(Nm)']))
    
    prediction=model.predict(t_d)
    
    # print(prediction)
    return str(prediction[0])

if __name__=="__main__":
    app.run(debug=True)
