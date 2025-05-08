import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('flight.pkl', 'rb'))

# Airplane name to numerical encoding
airplane_mapping = {
    "Airindia": 0,
    "Qatar Airways": 1,
    "Cessna": 2,
    "Emirates": 3,
    "Singapore Airlines": 4,
    "Lufthansa": 5,
    "Turkish Airlines": 6,
    "British Airways": 7,
    "American Airlines": 8,
    "Delta Airlines": 9,
    "United Airlines": 10,
    "Cathay Pacific": 11,
    "Qantas": 12,
    "Etihad Airways": 13,
    "Japan Airlines": 14,
    "Air France": 15,
    "KLM Royal Dutch Airlines": 16,
    "Swiss International Air Lines": 17,
    "Korean Air": 18,
    "China Airlines": 19,
    "Alaska Airlines": 20,
    "Hainan Airlines": 21,
    "Aeroflot": 22,
    "Garuda Indonesia": 23,
    "Thai Airways": 24,
    "Malaysia Airlines": 25,
    "Asiana Airlines": 26,
    "Eva Air": 27,
    "Air Canada": 28,
    "Aeromexico": 29,
    "LATAM Airlines": 30,
    "Saudi Airlines": 31,
    "Air New Zealand": 32,
    "South African Airways": 33,
    "Scoot": 34,
    "Indigo Airlines": 35,
    "SpiceJet": 36,
    "Go First": 37,
    "Vistara": 38,
    "Jet Airways": 39,
    "Ryanair": 40,
    "EasyJet": 41,
    "Wizz Air": 42,
    "Norwegian Air": 43,
    "Iberia": 44,
    "Air Europa": 45,
    "Finnair": 46,
    "SAS Scandinavian Airlines": 47,
    "Austrian Airlines": 48,
    "Air Portugal": 49,
    "Alitalia": 50,
    "Vietnam Airlines": 51,
    "Philippine Airlines": 52,
    "Bangkok Airways": 53,
    "SriLankan Airlines": 54,
    "Bamboo Airways": 55,
    "Azul Brazilian Airlines": 56,
    "Gol Transportes Aéreos": 57,
    "Virgin Atlantic": 58,
    "Virgin Australia": 59,
    "Peach Aviation": 60,
    "Tigerair": 61,
    "AirAsia": 62,
    "Lion Air": 63,
    "Jetstar Airways": 64,
    "Hawaiian Airlines": 65,
    "Fiji Airways": 66,
    "Edelweiss Air": 67,
    "Luxair": 68,
    "Brussels Airlines": 69,
    "WestJet": 70,
    "Air Transat": 71,
    "Sun Country Airlines": 72,
    "Frontier Airlines": 73,
    "Spirit Airlines": 74,
    "Allegiant Air": 75,
    "Volaris": 76,
    "Interjet": 77,
    "Copa Airlines": 78,
    "Avianca": 79,
    "LATAM Brazil": 80,
    "Azores Airlines": 81,
    "Air Baltic": 82,
    "Air Serbia": 83,
    "LOT Polish Airlines": 84,
    "Ukraine International Airlines": 85,
    "Czech Airlines": 86,
    "Croatia Airlines": 87,
    "Bulgarian Air": 88,
    "RwandAir": 89,
    "Ethiopian Airlines": 90,
    "Kenya Airways": 91,
    "EgyptAir": 92,
    "Royal Air Maroc": 93,
    "Tunisair": 94,
    "Air Mauritius": 95,
    "Air Seychelles": 96,
    "TAROM": 97,
    "S7 Airlines": 98,
    "Air Moldova": 99
}

source_mapping = {
    "Delhi": 0,
    "Mumbai": 1,
    "Bangalore": 2,
    "Chennai": 3,
    "Kolkata": 4,
    "Hyderabad": 5,
    "Ahmedabad": 6,
    "Pune": 7,
    "Jaipur": 8,
    "Lucknow": 9,
    "Kochi": 10,
    "Chandigarh": 11,
    "Indore": 12,
    "Surat": 13,
    "Nagpur": 14,
    "Visakhapatnam": 15,
    "Goa": 16,
    "Varanasi": 17,
    "Coimbatore": 18,
    "Madurai": 19,
    "Rajkot": 20,
    "Bhopal": 21,
    "Mysuru": 22,
    "Vadodara": 23,
    "Noida": 24,
    "Patna": 25,
    "Shillong": 26,
    "Agra": 27,
    "Ranchi": 28,
    "Trivandrum": 29,
    "Guwahati": 30,
    "Kanpur": 31,
    "Meerut": 32,
    "Kolkata": 33,
    "Faridabad": 34,
    "Tirupati": 35,
    "Gurgaon": 36,
    "Bhubaneswar": 37,
    "Nashik": 38,
    "Bikaner": 39,
    "Dehradun": 40,
    "Jammu": 41,
    "Udaipur": 42,
    "Amritsar": 43,
    "Bardhaman": 44,
    "Raipur": 45,
    "Patiala": 46,
    "Gaya": 47,
    "Haldwani": 48,
    "Bardoli": 49
}

destination_mapping = {
    "Delhi": 0,
    "Mumbai": 1,
    "Bangalore": 2,
    "Chennai": 3,
    "Kolkata": 4,
    "Hyderabad": 5,
    "Ahmedabad": 6,
    "Pune": 7,
    "Jaipur": 8,
    "Lucknow": 9,
    "Kochi": 10,
    "Chandigarh": 11,
    "Indore": 12,
    "Surat": 13,
    "Nagpur": 14,
    "Visakhapatnam": 15,
    "Goa": 16,
    "Varanasi": 17,
    "Coimbatore": 18,
    "Madurai": 19,
    "Rajkot": 20,
    "Bhopal": 21,
    "Mysuru": 22,
    "Vadodara": 23,
    "Noida": 24,
    "Patna": 25,
    "Shillong": 26,
    "Agra": 27,
    "Ranchi": 28,
    "Trivandrum": 29,
    "Guwahati": 30,
    "Kanpur": 31,
    "Meerut": 32,
    "Tirupati": 33,
    "Gurgaon": 34,
    "Bhubaneswar": 35,
    "Nashik": 36,
    "Bikaner": 37,
    "Dehradun": 38,
    "Jammu": 39,
    "Udaipur": 40,
    "Amritsar": 41,
    "Bardhaman": 42,
    "Raipur": 43,
    "Patiala": 44,
    "Gaya": 45,
    "Haldwani": 46,
    "Bardoli": 47,
    "Firozabad": 48,
    "Muzaffarpur": 49,
    "Jalandhar": 50,
    "Sonipat": 51,
    "Aligarh": 52,
    "Dhanbad": 53,
    "Kota": 54,
    "Satna": 55,
    "Durg": 56,
    "Shahjahanpur": 57,
    "Jhansi": 58,
    "Karnal": 59,
    "Srinagar": 60,
    "Bilaspur": 61,
    "Kollam": 62,
    "Chhapra": 63,
    "Raurkela": 64,
    "Jammu": 65,
    "Bhubaneshwar": 66,
    "Gwalior": 67,
    "Jorhat": 68,
    "Agartala": 69,
    "Mangalore": 70,
    "Raigarh": 71,
    "Tirunelveli": 72,
    "Ambala": 73,
    "Kurukshetra": 74,
    "Muzaffarnagar": 75,
    "Moradabad": 76,
    "Puducherry": 77,
    "Bhagalpur": 78,
    "Hoshiarpur": 79,
    "Faridkot": 80,
    "Siliguri": 81,
    "Palakkad": 82,
    "Jammu": 83,
    "Nanded": 84,
    "Mysore": 85,
    "Gurugram": 86,
    "Muzaffarpur": 87,
    "Hassan": 88,
    "Rishikesh": 89,
    "Chandrapur": 90,
    "Kanpur": 91,
    "Nagapattinam": 92,
    "Srinagar": 93,
    "Bilaspur": 94,
    "Karnal": 95,
    "Aligarh": 96,
    "Ambala": 97,
    "Shimla": 98,
    "Surat": 99
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        airline_name = request.form["AIRLINE"]
        source = request.form["Source"]
        destination = request.form["Destination"]
        total_stops = int(request.form["Total_stops"])
        dep_hours = int(request.form["Dep_hours"])
        arrival_hours = int(request.form["Arrival_hours"])
        duration_hours = int(request.form["Duration_hours"])

        if airline_name in airplane_mapping:
            airline = airplane_mapping[airline_name]
        else:
            return render_template('index.html', error=f"Airline '{airline_name}' not recognized.")
        
        if source in source_mapping:
            source = source_mapping[source]
        else:
            return render_template('index.html', error=f"Source city '{source}' not recognized.")
        
        if destination in destination_mapping:
            destination = destination_mapping[destination]
        else:
            return render_template('index.html', error=f"Destination city '{destination}' not recognized.")
        
        input_features = np.array([[airline, source, destination, total_stops, dep_hours, arrival_hours, duration_hours]])

        prediction = model.predict(input_features)
        output = prediction[0]

        return render_template('index.html', prediction=f"Predicted Price: {output}")

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
