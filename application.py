import os
import datetime

from flask import Flask, flash, redirect, render_template, request, session, jsonify
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions
from werkzeug.security import check_password_hash, generate_password_hash

from dotenv import load_dotenv
import forecast as fc

import sqlite3

def create_app():
    load_dotenv()

    conn = sqlite3.connect('alerts.db',check_same_thread=False)
    db = conn.cursor()

    # Configure application
    app = Flask(__name__)

    # Ensure templates are auto-reloaded
    app.config["TEMPLATES_AUTO_RELOAD"] = True


    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response

    @app.route("/")

    def menu():
        return render_template("menu.html")



    @app.route("/commonalerts", methods=["GET", "POST"])

    def commonalerts():

        if request.method == "POST":
            if not request.form.get("calamity"):
                return "Must enter the calamity."
            elif not request.form.get("location"):
                return "Must enter the location"

            c = request.form.get("calamity")
            l = request.form.get("location")
            d = request.form.get("description")
            s = {"success":"yes"}
            db.execute("INSERT INTO commonalerts (calamity,location,description) VALUES (?,?,?)",
                           (c,l,d))
            conn.commit()
            return render_template("noflood.html"),200
        else:
            return render_template("alerts.html")

    @app.route("/getcommonalerts")

    def getcommonalerts():
    
        alerts=[]
        w = db.execute("SELECT * FROM commonalerts ORDER BY id DESC ")
        for w1 in w:
            s = {"datetime":w1[1],"location":w1[3],"calamity":w1[2],"description":w1[4]}
            alerts.append(s)

        return jsonify(alerts)
    
    @app.route("/contact")
    def contact():
        return render_template("contact.html")

    @app.route("/govtalerts", methods=["GET", "POST"])

    def govtalerts():

        if request.method == "POST":
            if not request.form.get("username"):
                return "Must enter a username"
            elif not request.form.get("password"):
                return "Must enter the password"
            elif not request.form.get("calamity"):
                return "Must enter the calamity"
            elif not request.form.get("location"):
                return "Must enter the location"
            elif not request.form.get("description"):
                return "Must enter the description"

            username=request.form.get("username")
            password=request.form.get("password")
            rows = db.execute("SELECT * FROM govtids WHERE username = ?",(username,))
            row = db.fetchone()
            if row is None or not check_password_hash(row[2], request.form.get("password")):
                return "Invalid username and/or password"


            c = request.form.get("calamity")
            l = request.form.get("location")
            d = request.form.get("description")
            s = {"success":"yes"}

            db.execute("INSERT INTO govtalerts (calamity,location,description) VALUES (?,?,?)",
                           (c,l,d))
            conn.commit()
            return render_template("noflood.html"),200
        else:
            return render_template("govtalerts.html")

    @app.route("/getgovtalerts")

    def getgovtalerts():

        alerts=[]
        w = db.execute("SELECT * FROM govtalerts ORDER BY id DESC ")
        for w1 in w:
            s = {"datetime":w1[1],"location":w1[3],"calamity":w1[2],"description":w1[4]}
            alerts.append(s)
        return jsonify(alerts)
    
    @app.route('/send-location', methods=['POST'])
    def send_location():
       data = request.get_json()
       latitude = data.get('latitude')
       longitude = data.get('longitude')
    # Handle the received location (e.g., store in database or send to emergency services)
    return jsonify({"status": "success"}), 200
    
    @app.route("/aialerts", methods=["GET","POST"])

    def aialerts():
        if request.method=="POST":
            data = fc.getData()
            prob = fc.forecast(data)
            if prob>0.5:
                c="Flood"
                l="Pune"
                d="Flood Warning! Reach safe places at high grounds ASAP."
                s = {"success":"yes"}
                db.execute("INSERT INTO govtalerts (calamity,location,description) VALUES (?,?,?)",
                           (c,l,d))
                conn.commit()
                return render_template("disaster.html")
            else:
                return render_template("noflood.html")
        else:
            return render_template("aipredict.html")


    #@app.route("/generateids")

    # def generateids():
    #     usernames=["earthquakeagencyofindia","hurricaneagencyofindia","floodagencyofindia","disastermanagementagencyofindia"]
    #     passwords=["eaoi","haoi","faoi","dmaoi"]

    #     for i in range(4):
    #         db.execute("INSERT INTO govtids (username,password) VALUES (?,?)",(usernames[i],generate_password_hash(passwords[i])))

    @app.route("/viewgovtalerts")

    def viewgovtalerts():

        alerts=[]
        w = db.execute("SELECT * FROM govtalerts ORDER BY id DESC ")
        for w1 in w:
            s = {"datetime":w1[1],"location":w1[3],"calamity":w1[2],"description":w1[4]}
            alerts.append(s)
        return render_template("view.html",rows=alerts,alert="Govt Alerts")

    @app.route("/viewcommonalerts")

    def viewcommonalerts():

        alerts=[]
        w = db.execute("SELECT * FROM commonalerts ORDER BY id DESC ")
        for w1 in w:
            s = {"datetime":w1[1],"location":w1[3],"calamity":w1[2],"description":w1[4]}
            alerts.append(s)
        return render_template("view.html",rows=alerts,alert="Common Alerts")

    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
