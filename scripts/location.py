# import json
# from urllib.request import urlopen

# def get_location():
#     # Fallback to another service
#     url = 'http://ip-api.com/json'
#     response = urlopen(url)
#     data = json.load(response)
    
#     return data

# data = get_location()
# print(data)



import requests
import json

send_url = "http://api.ipstack.com/check?access_key=110ee64e48f3e793742471c93a495b93"
geo_req = requests.get(send_url)
geo_json = json.loads(geo_req.text)
latitude = geo_json['latitude']
longitude = geo_json['longitude']
city = geo_json['city']

print(latitude, longitude, city)
