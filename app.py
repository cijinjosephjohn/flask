from flask import Flask, render_template,request
import pickle
import numpy as np
import dill
import matplotlib.pyplot as plt


def fun1(ans):
    if ans =="Male":
        return 1
    else:
        return 0
def fun5(ans):
    if ans =="Yes":
        return 1
    else:
        return 0

def fun2(ans):
    ans =int(ans)
    if ans ==0:
        return 0
    elif ans ==1:
        return 1
    elif ans ==2:
        return 2
    else:
        return 3
def fun6(ans):
    if ans =="Graduate":
        return 1
    else:
        return 0
def fun3(ans):
    if ans =="Urban":
        return 1
    elif ans =="Semiurban":
        return 3
    else:
        return 2

model = pickle.load(open('models/model8605.pkl','rb'))
explainer = dill.load(open('models/explainer.pkl','rb'))



app = Flask(__name__,template_folder='templates')

@app.route("/")
def file():
    return render_template("index.html")

@app.route('/submit',methods=['POST'])
def index():
    gen= request.form['gender']
    gen=fun1(gen)
    mar = request.form['married']
    mar = fun5(mar)
    dep=request.form['dependence']
    dep = fun2(dep)
    app_in = request.form['applicant_income']
    app_in = int(app_in)
    app_in_cpy =np.sqrt(app_in)
    co_in = request.form['coapplicant_income']
    co_in =int(co_in)
    co_in_cpy = np.sqrt(co_in)
    loan_amt=request.form['loan_amount']
    loan_amt = int(loan_amt)
    loan_amt_cpy = np.sqrt(loan_amt)
    loan_amt_term=request.form['loan_amount_term']
    loan_amt_term=int(loan_amt_term)
    crd_hsty = request.form['credit_history']
    crd_hsty=int(crd_hsty)
    prpty_area = request.form['property_area']
    prpty_area=fun3(prpty_area)
    educated=request.form['educated']
    educated = fun6(educated)
    self_emp=request.form['self_employed']
    self_emp=fun5(self_emp)
    print(gen,mar,dep,app_in,co_in,loan_amt,loan_amt_term,crd_hsty,prpty_area,educated,self_emp)
    pred = model.predict([[gen,mar,dep,educated,self_emp,app_in_cpy,co_in_cpy,loan_amt_cpy,loan_amt_term,crd_hsty,prpty_area]])
    predict_rf = lambda x : model.predict_proba(x).astype(float)
    arr = np.array([gen,mar,dep,educated,self_emp,app_in_cpy,co_in_cpy,loan_amt_cpy,loan_amt_term,crd_hsty,prpty_area])
    exp = explainer.explain_instance(arr,predict_rf,num_features=7)
    exp_img =exp.as_pyplot_figure()
    # fig =exp.get_figure()
    # exp.show_in_notebook(show_table=True, show_all=False)
    
    # with open("static/explanation.png","wb") as f:
    #     f.write(exp_img)
    exp_img.set_size_inches(20,10)
    
    exp_img.savefig("static/explanation.png")
    print(pred)
    return render_template('result.html',data=pred, image_path="static/explanation.png")

if __name__=="__main__":
    app.run(debug=True)