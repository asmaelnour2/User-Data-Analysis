import requests
import pandas as pd

limit = 30
total = 0
skip = 0
listpd = []
page = 0

while True:
    url = f'https://dummyjson.com/users?limit={limit}&skip={skip}'
    response = requests.get(url)
    data = response.json()['users']
    listpd.append(pd.DataFrame(data))
    total = response.json()['total']
    skip+=limit
    if skip >= total:
        break 
    print(page)
    page+=1

users = pd.concat(listpd)
print(users)
    
users.to_csv('users.csv', index=False)
print("Saved to users.xlsx")
    